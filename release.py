#!/usr/bin/env python3
"""
cattle-auction release script.

Updates version metadata and CHANGELOG.md, runs tests, creates one release
commit containing all intended repo changes, tags that commit, pushes main and
the tag, and publishes a GitHub release via the gh CLI.

Usage:
    python3 release.py                 # auto-detect version from commits/dirty tree
    python3 release.py 1.2.0           # explicit version
    python3 release.py --dry-run       # preview without making changes
    python3 release.py 1.2.0 --dry-run
    python3 release.py --no-github     # tag only, skip GitHub release
    python3 release.py --no-tests      # skip pytest verification
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

REMOTE = "origin"
BRANCH = "main"
REPO = "rteoo/cattle-auction"
CHANGELOG = "CHANGELOG.md"
PYPROJECT = "pyproject.toml"
GIT_AUTHOR = "rteoo <rteoo@users.noreply.github.com>"
TEST_CMD = ["uv", "run", "pytest", "tests/", "-q"]

STAGE_DIRS = ["bench/", "pipeline/", "models/", "prompts/", "tests/", ".github/"]
STAGE_FILES = [
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "CHANGELOG.md",
    "LICENSE",
    "README.md",
    "benchmark.py",
    "main.py",
    "pyproject.toml",
    "release.py",
    "uv.lock",
]
PACKAGE_FILES = {"main.py", "prompts/extraction.txt", "prompts/metadata.txt", "prompts/verify.txt"}

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "qwen3.5:2b-q4_K_M"

# ── Colors (ANSI) ───────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def error(msg):
    print(f"{RED}✗ Error: {msg}{NC}", file=sys.stderr)


def success(msg):
    print(f"{GREEN}✓ {msg}{NC}")


def info(msg):
    print(f"{YELLOW}ℹ {msg}{NC}")


def step(msg):
    print(f"\n{YELLOW}→ {msg}{NC}")


def dim(msg):
    print(f"{CYAN}  {msg}{NC}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_env():
    """Return env dict with correct git author/committer identity."""
    import os
    env = os.environ.copy()
    m = re.match(r"^(.+?)\s*<(.+?)>$", GIT_AUTHOR)
    if m:
        name, email = m.group(1), m.group(2)
        env["GIT_AUTHOR_NAME"] = name
        env["GIT_AUTHOR_EMAIL"] = email
        env["GIT_COMMITTER_NAME"] = name
        env["GIT_COMMITTER_EMAIL"] = email
    return env


def run(cmd, capture=True, check=True):
    """Run a shell command, return stdout stripped."""
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    if capture:
        return result.stdout.strip()
    return ""


def run_ok(cmd):
    """Run a command and return True if it exits 0."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def git(*args):
    return run(["git", *args])


def git_env(*args):
    """Run git with the correct author/committer identity."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
        env=_git_env(),
    )
    return result.stdout.strip()


def git_ok(*args):
    return run_ok(["git", *args])


# ── Parse args ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a cattle-auction release: tag, changelog, GitHub release.",
    )
    parser.add_argument(
        "version", nargs="?", default=None,
        help="Semver version (e.g. 1.2.0 or v1.2.0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="Preview without making changes.",
    )
    parser.add_argument(
        "--no-github", action="store_true",
        help="Tag only, skip GitHub release creation.",
    )
    parser.add_argument(
        "--no-tests", action="store_true",
        help="Skip the release test command.",
    )
    parser.add_argument(
        "--commit-message", default=None,
        help="Commit message for the release commit. Defaults to 'chore: release vX.Y.Z'.",
    )
    return parser.parse_args()


# ── Prerequisites ────────────────────────────────────────────────────────────

def check_prerequisites(skip_github):
    step("Checking prerequisites...")

    if not shutil.which("git"):
        error("git not found")
        sys.exit(1)

    gh = None
    if not skip_github:
        gh = shutil.which("gh")
        if not gh:
            error("gh CLI not found — install from https://cli.github.com/")
            sys.exit(1)
        if not run_ok([gh, "auth", "status"]):
            info("gh auth status failed; continuing because gh release create may still work via available credentials")
        else:
            success("gh CLI authenticated")

    return gh


# ── Working tree helpers ─────────────────────────────────────────────────────

def has_pending_changes():
    return bool(git("status", "--porcelain"))


def is_release_path_allowed(path):
    """Return whether a path belongs to the release staging allowlist."""
    candidate = Path(path)
    allowed_files = {Path(file_path) for file_path in STAGE_FILES}
    if candidate in allowed_files:
        return True
    return any(
        candidate == Path(directory) or Path(directory) in candidate.parents
        for directory in STAGE_DIRS
    )


def verify_staged_release_paths():
    """Fail before commit if pre-staged files fall outside the release scope."""
    staged = git("diff", "--cached", "--name-only")
    disallowed = [path for path in staged.splitlines() if path and not is_release_path_allowed(path)]
    if disallowed:
        error("Staged paths outside the release allowlist: " + ", ".join(disallowed))
        sys.exit(1)


def verify_package_artifacts(build_dir):
    wheels = sorted(Path(build_dir).glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {len(wheels)}")

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
    missing = sorted(PACKAGE_FILES - names)
    if missing:
        raise RuntimeError("wheel is missing: " + ", ".join(missing))

    sdists = sorted(Path(build_dir).glob("*.tar.gz"))
    if len(sdists) != 1:
        raise RuntimeError(f"expected one source archive, found {len(sdists)}")

    import tarfile
    with tarfile.open(sdists[0]) as sdist:
        leaked = [
            name for name in sdist.getnames()
            if any(name.split("/", 1)[-1].startswith(prefix) for prefix in (".claude", ".clawpatch"))
        ]
    if leaked:
        raise RuntimeError("source archive includes local tooling: " + ", ".join(leaked))


def run_package_build(dry_run):
    step("Building release artifacts...")
    if dry_run:
        info("Would run uv build and verify the wheel CLI/prompts and source archive scope")
        return
    if not shutil.which("uv"):
        error("uv not found; cannot verify release artifacts")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="cattle-auction-build-") as build_dir:
        run(["uv", "build", "--out-dir", build_dir], capture=False)
        try:
            verify_package_artifacts(build_dir)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            error(f"Release artifact verification failed: {exc}")
            sys.exit(1)
    success("Release artifacts verified")


# ── Stage and commit release changes ─────────────────────────────────────────

def stage_release_changes(dry_run):
    step("Staging release changes...")
    pending = git("status", "--short")
    if pending:
        print(pending)
    else:
        info("Working tree is clean before release metadata updates")

    if dry_run:
        info("Would stage only allowlisted source/docs files")
        return

    # Stage only allowlisted paths. In particular, do not use `git add -u`,
    # which would pull unrelated tracked changes into the release commit.
    for d in STAGE_DIRS:
        if Path(d).is_dir():
            try:
                git("add", "-A", "--", d)
            except subprocess.CalledProcessError:
                pass

    for f in STAGE_FILES:
        try:
            git("add", "-A", "--", f)
        except subprocess.CalledProcessError:
            pass

    verify_staged_release_paths()


def commit_release_changes(tag, message, dry_run):
    step("Creating release commit...")

    if dry_run:
        info(f"Would commit staged changes with message: {message or f'chore: release {tag}'}")
        return

    if git_ok("diff", "--cached", "--quiet"):
        error("No staged release changes to commit")
        sys.exit(1)

    git_env("commit", "-m", message or f"chore: release {tag}")
    success("Release commit created")


# ── Version detection ────────────────────────────────────────────────────────

def detect_version(explicit_version):
    step("Determining version...")

    try:
        last_tag = git("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        last_tag = "v0.0.0"

    info(f"Last tag: {last_tag}")

    bare = last_tag.lstrip("v")
    parts = bare.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0

    pending = has_pending_changes()
    try:
        all_commits = git("log", f"{last_tag}..HEAD", "--oneline")
    except subprocess.CalledProcessError:
        all_commits = git("log", "--oneline")

    commit_lines = [l for l in all_commits.splitlines() if l.strip()] if all_commits else []

    # Filter out release-machinery commits
    real_commits = [
        l for l in commit_lines
        if not re.match(
            r"^[a-f0-9]+ chore: (release changes|bump version to |docs: update CHANGELOG)",
            l,
        )
    ]

    info(f"Commits since {last_tag}: {len(commit_lines)} ({len(real_commits)} meaningful)")
    for line in commit_lines:
        dim(line)
    if pending:
        info("Working tree has pending changes that will be included in the release commit")

    if not real_commits and not pending:
        error(f"Only release-machinery commits since {last_tag} — nothing meaningful to release")
        sys.exit(1)

    if explicit_version is None:
        subjects = "\n".join(real_commits)
        if re.search(r"^[a-f0-9]+ feat(\([^)]+\))?(!)?: ", subjects, re.MULTILINE):
            bump = "minor"
            version = f"{major}.{minor + 1}.0"
        else:
            bump = "patch"
            version = f"{major}.{minor}.{patch + 1}"
        info(f"Bump type: {bump} → {version}")
    else:
        version = explicit_version.lstrip("v")

    if not re.match(r"^\d+\.\d+\.\d+$", version):
        error(f"Invalid version: {version} (expected X.Y.Z)")
        sys.exit(1)

    tag = f"v{version}"

    if git_ok("rev-parse", tag):
        error(f"Tag '{tag}' already exists")
        sys.exit(1)

    success(f"Version: {tag}")
    if pending and not real_commits:
        real_commits = ["worktree: pending release changes"]

    return version, tag, last_tag, commit_lines, real_commits


# ── Update pyproject.toml ───────────────────────────────────────────────────

def update_pyproject(version, dry_run):
    step(f"Updating {PYPROJECT} to {version}...")

    if dry_run:
        info(f"Would update {PYPROJECT} version to {version}")
        return

    path = Path(PYPROJECT)
    content = path.read_text(encoding="utf-8")
    updated = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(updated, encoding="utf-8")

    if content == updated:
        info(f"Version already at {version}, no changes needed")
    else:
        success(f"{PYPROJECT} updated")


# ── Build release notes ─────────────────────────────────────────────────────

def build_release_notes(version, tag, last_tag, commit_lines, real_commits):
    step("Building release notes...")

    # 1. Try CHANGELOG
    changelog_path = Path(CHANGELOG)
    if changelog_path.exists():
        content = changelog_path.read_text(encoding="utf-8")
        pattern = rf"## \[{re.escape(version)}\].*?\n(.*?)(?=\n## \[|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            entry = match.group(1).strip()
            if entry:
                success("Release notes from CHANGELOG")
                return entry

    # 2. Generate from commits
    info("No CHANGELOG entry found — generating from commits")

    subjects = []
    for line in real_commits:
        parts = line.split(" ", 1)
        subjects.append(parts[1] if len(parts) > 1 else line)

    def fmt_line(raw):
        """Format a conventional commit line."""
        m = re.match(r"^([a-z]+)(?:\(([^)]+)\))?(!)?: (.+)", raw)
        if m:
            scope = m.group(2)
            msg = m.group(4)
            msg = msg[0].upper() + msg[1:] if msg else msg
            if scope:
                return f"- {msg} (`{scope}`)"
            return f"- {msg}"
        return f"- {raw}"

    def build_section(header, items):
        if not items:
            return ""
        lines = [f"### {header}"]
        for item in items:
            lines.append(fmt_line(item))
        return "\n".join(lines) + "\n"

    feats = [s for s in subjects if re.match(r"^feat(\([^)]+\))?(!)?: ", s)]
    fixes = [s for s in subjects if re.match(r"^fix(\([^)]+\))?: ", s)]
    perf = [s for s in subjects if re.match(r"^perf(\([^)]+\))?: ", s)]
    refactor = [s for s in subjects if re.match(r"^refactor(\([^)]+\))?: ", s)]
    docs = [s for s in subjects if re.match(r"^docs(\([^)]+\))?: ", s)]
    chore = [s for s in subjects if re.match(r"^chore(\([^)]+\))?: ", s)]
    other_pattern = r"^(feat|fix|perf|refactor|style|docs|test|chore|ci|build)(\([^)]+\))?(!)?: "
    other = [s for s in subjects if not re.match(other_pattern, s)]

    notes = ""
    notes += build_section("New Features", feats)
    notes += build_section("Bug Fixes", fixes)
    notes += build_section("Performance", perf)
    notes += build_section("Improvements", refactor)
    notes += build_section("Documentation", docs)
    notes += build_section("Maintenance", chore)
    notes += build_section("Other", other)

    if not notes.strip():
        notes = "### Changes\n"
        for s in subjects:
            notes += f"- {s}\n"

    # 3. Generate human-readable summary via local Ollama
    human_summary = _generate_ollama_summary(tag, last_tag, subjects)

    if human_summary:
        notes = f"{human_summary}\n\n{notes}"

    success("Release notes ready:")
    for line in notes.strip().splitlines()[:20]:
        print(f"  {line}")
    if len(notes.strip().splitlines()) > 20:
        dim("...")

    return notes.strip()


def _generate_ollama_summary(tag, last_tag, subjects):
    """Generate a human-readable release summary using local Ollama. Returns None on failure."""

    # Check if Ollama is reachable
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except (urllib.error.URLError, OSError):
        info(f"Ollama not available at {OLLAMA_URL} — skipping summary")
        return None

    info(f"Generating release summary with Ollama ({OLLAMA_MODEL})...")

    # Collect diff stats for context
    try:
        diff_stat = git("diff", f"{last_tag}..HEAD", "--stat")
    except subprocess.CalledProcessError:
        diff_stat = "(unavailable)"

    prompt = (
        "You are writing a release summary for a cattle auction video pipeline CLI "
        "(Python, yt-dlp, Whisper, OCR, LLM extraction).\n\n"
        f"Version: {tag} (previous: {last_tag})\n\n"
        f"Commits:\n{chr(10).join(subjects)}\n\n"
        f"Files changed:\n{diff_stat}\n\n"
        "Write a concise 1-3 sentence summary of what changed in this release "
        "and why it matters to the user. Focus on the user-facing impact, not "
        "implementation details. Write in English. No markdown formatting, just "
        "plain text. No thinking tags."
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    })

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Strip <think>...</think> tags if present
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        if text:
            success("Summary generated")
            return text
        else:
            info("Ollama returned empty response — skipping summary")
            return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        info(f"Ollama request failed ({exc}) — skipping summary")
        return None


# ── Update CHANGELOG.md ─────────────────────────────────────────────────────

def update_changelog(version, release_notes, dry_run):
    step("Updating CHANGELOG.md...")

    if dry_run:
        info("Would prepend entry to CHANGELOG.md")
        return

    today = date.today().isoformat()
    entry = f"## [{version}] — {today}\n\n{release_notes}"

    path = Path(CHANGELOG)
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if re.search(rf"^## \[{re.escape(version)}\]\b", content, re.MULTILINE):
            info(f"CHANGELOG.md already has an entry for {version}")
            return
        lines = content.split("\n", 1)
        if lines[0].startswith("# "):
            header = lines[0]
            body = lines[1] if len(lines) > 1 else ""
            updated = f"{header}\n\n{entry}\n{body}"
        else:
            updated = f"{entry}\n\n{content}"
    else:
        updated = f"# Changelog\n\n{entry}\n"

    path.write_text(updated, encoding="utf-8")
    success("CHANGELOG.md updated")


# ── Tests ───────────────────────────────────────────────────────────────────

def run_tests(skip_tests, dry_run):
    step("Running release tests...")

    if skip_tests:
        info("Skipping tests by request")
        return
    if dry_run:
        info("Would run: " + " ".join(TEST_CMD))
        return
    if not shutil.which(TEST_CMD[0]):
        error(f"{TEST_CMD[0]} not found; cannot run release tests")
        sys.exit(1)

    run(TEST_CMD, capture=False)
    success("Tests passed")


# ── Tag ──────────────────────────────────────────────────────────────────────

def create_tag(tag, dry_run):
    step(f"Creating tag {tag}...")

    if dry_run:
        info(f'Would create: git tag -a {tag} -m "Release {tag}"')
        return

    git_env("tag", "-a", tag, "-m", f"Release {tag}")
    success(f"Tag {tag} created")


# ── Push ─────────────────────────────────────────────────────────────────────

def push(tag, dry_run):
    step(f"Pushing to {REMOTE}...")

    if dry_run:
        info(f"Would push branch {BRANCH} and tag {tag}")
        return

    git("push", REMOTE, BRANCH)
    success(f"Pushed branch {BRANCH}")
    git("push", REMOTE, tag)
    success(f"Pushed tag {tag}")


# ── GitHub release ───────────────────────────────────────────────────────────

def create_github_release(tag, release_notes, gh, dry_run):
    step(f"Creating GitHub release {tag}...")

    if dry_run:
        info(f"Would create GitHub release: {tag}")
        return

    import tempfile
    notes_file = Path(tempfile.mktemp(suffix=".md"))
    notes_file.write_text(release_notes, encoding="utf-8")

    try:
        run(
            [gh, "release", "create", tag,
             "--repo", REPO,
             "--title", tag,
             "--notes-file", str(notes_file)],
            capture=False,
        )
        success("GitHub release published")
    except subprocess.CalledProcessError:
        error("Failed to create GitHub release")
        info(f"Create manually: https://github.com/{REPO}/releases/new?tag={tag}")
        sys.exit(1)
    finally:
        notes_file.unlink(missing_ok=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    gh = check_prerequisites(args.no_github)
    version, tag, last_tag, commit_lines, real_commits = detect_version(args.version)
    update_pyproject(version, args.dry_run)

    release_notes = build_release_notes(version, tag, last_tag, commit_lines, real_commits)

    update_changelog(version, release_notes, args.dry_run)
    run_tests(args.no_tests, args.dry_run)
    run_package_build(args.dry_run)
    stage_release_changes(args.dry_run)
    commit_release_changes(tag, args.commit_message, args.dry_run)
    create_tag(tag, args.dry_run)
    push(tag, args.dry_run)

    if not args.no_github:
        create_github_release(tag, release_notes, gh, args.dry_run)

    print()
    if args.dry_run:
        success(f"Dry run complete for {tag}")
    else:
        success(f"Released {tag}")
    if not args.no_github and not args.dry_run:
        print(f"  https://github.com/{REPO}/releases/tag/{tag}")


if __name__ == "__main__":
    main()
