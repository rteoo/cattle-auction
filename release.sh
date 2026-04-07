#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

REMOTE="origin"
BRANCH="main"
REPO="TeodoroRodrigo/cattle-auction"
CHANGELOG="CHANGELOG.md"

print_error()   { echo -e "${RED}✗ Error: $1${NC}" >&2; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info()    { echo -e "${YELLOW}ℹ $1${NC}"; }
print_step()    { echo -e "\n${YELLOW}→ $1${NC}"; }
print_dim()     { echo -e "${CYAN}  $1${NC}"; }

usage() {
    cat << EOF
Usage: $0 [version] [OPTIONS]

Create a git tag and publish a GitHub release for cattle-auction.

If version is omitted, it is auto-determined from the last tag and commit
history: any commit with "feat:" bumps the minor version; otherwise patch.

Arguments:
  version           Optional semver override, e.g. 1.3.0 or v1.3.0

Options:
  -h, --help        Show this help
  -d, --dry-run     Preview without making changes
  --no-github       Tag only, skip GitHub release creation

Examples:
  $0                  # auto-detect version from commits
  $0 1.3.0            # explicit version
  $0 --dry-run        # preview auto-detected version
  $0 1.3.0 --dry-run

EOF
    exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────────

VERSION=""
DRY_RUN=false
SKIP_GITHUB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)    usage ;;
        -d|--dry-run) DRY_RUN=true; shift ;;
        --no-github)  SKIP_GITHUB=true; shift ;;
        v*|[0-9]*)    VERSION=$1; shift ;;
        *)            print_error "Unknown option: $1"; usage ;;
    esac
done

# ── Prerequisites ─────────────────────────────────────────────────────────────

print_step "Checking prerequisites..."

GH_CMD=""
if [ "$SKIP_GITHUB" != "true" ]; then
    for candidate in gh /opt/homebrew/bin/gh /usr/local/bin/gh; do
        if command -v "$candidate" &>/dev/null 2>&1; then
            GH_CMD="$candidate"; break
        fi
    done
    if [ -z "$GH_CMD" ]; then
        print_error "gh CLI not found — install from https://cli.github.com/"
        exit 1
    fi
    if ! $GH_CMD auth status &>/dev/null; then
        print_error "gh not authenticated — run: gh auth login"
        exit 1
    fi
    print_success "gh CLI authenticated"
fi

# ── Commit pending changes first ──────────────────────────────────────────────

PENDING=$(PATH=/opt/homebrew/bin:$PATH git status --porcelain)
if [ -n "$PENDING" ]; then
    print_step "Committing pending changes..."
    PATH=/opt/homebrew/bin:$PATH git status --short

    if [ "$DRY_RUN" = "true" ]; then
        print_info "Would commit all tracked changes"
    else
        # Stage tracked files only — new files must be explicitly added.
        # This avoids accidentally committing secrets, PDFs, or other untracked files.
        PATH=/opt/homebrew/bin:$PATH git add -u

        # Also stage new files in well-known source directories (not data/output)
        for dir in pipeline/ models/ prompts/ tests/ .github/; do
            if [ -d "$dir" ]; then
                PATH=/opt/homebrew/bin:$PATH git add "$dir" 2>/dev/null || true
            fi
        done
        # Stage root config files
        PATH=/opt/homebrew/bin:$PATH git add .gitignore pyproject.toml \
            CLAUDE.md release.sh main.py 2>/dev/null || true

        # Check if anything is staged
        if PATH=/opt/homebrew/bin:$PATH git diff --cached --quiet; then
            print_info "No relevant changes to commit"
        else
            PATH=/opt/homebrew/bin:$PATH git commit -m "chore: pre-release changes"
            print_success "Changes committed"
        fi
    fi
fi

# ── Auto-determine version ────────────────────────────────────────────────────

print_step "Determining version..."

LAST_TAG=$(PATH=/opt/homebrew/bin:$PATH git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
print_info "Last tag: $LAST_TAG"

# Parse last tag into major.minor.patch
LAST_BARE="${LAST_TAG#v}"
MAJOR=$(echo "$LAST_BARE" | cut -d. -f1)
MINOR=$(echo "$LAST_BARE" | cut -d. -f2)
PATCH=$(echo "$LAST_BARE" | cut -d. -f3)

# Commits since last tag — EXCLUDE release-machinery commits
ALL_COMMITS=$(PATH=/opt/homebrew/bin:$PATH git log "${LAST_TAG}..HEAD" --oneline)
REAL_COMMITS=$(echo "$ALL_COMMITS" | grep -vE "^[a-f0-9]+ chore: (release changes|pre-release changes|bump version to |docs: update CHANGELOG)" || true)
COMMIT_COUNT=$(echo "$ALL_COMMITS" | grep -c . || true)
REAL_COUNT=$(echo "$REAL_COMMITS" | grep -c . || true)

if [ "$COMMIT_COUNT" -eq 0 ]; then
    print_error "No commits since $LAST_TAG — nothing to release"
    exit 1
fi

print_info "Commits since $LAST_TAG: $COMMIT_COUNT ($REAL_COUNT meaningful)"
echo "$ALL_COMMITS" | while read -r line; do print_dim "$line"; done

if [ "$REAL_COUNT" -eq 0 ]; then
    print_error "Only release-machinery commits since $LAST_TAG — nothing meaningful to release"
    exit 1
fi

if [ -z "$VERSION" ]; then
    if echo "$REAL_COMMITS" | grep -qE "^[a-f0-9]+ feat(\([^)]+\))?(!)?:"; then
        BUMP_TYPE="minor"
        AUTO_VERSION="${MAJOR}.$((MINOR + 1)).0"
    else
        BUMP_TYPE="patch"
        AUTO_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
    fi
    print_info "Bump type: $BUMP_TYPE → $AUTO_VERSION"
    VERSION="$AUTO_VERSION"
fi

# Normalize version
VERSION_BARE="${VERSION#v}"
VERSION_TAG="v${VERSION_BARE}"

if ! [[ "$VERSION_BARE" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version: $VERSION_BARE (expected X.Y.Z)"
    exit 1
fi

if PATH=/opt/homebrew/bin:$PATH git rev-parse "$VERSION_TAG" &>/dev/null; then
    print_error "Tag '$VERSION_TAG' already exists"
    exit 1
fi

print_success "Version: $VERSION_TAG"

# ── Update pyproject.toml ────────────────────────────────────────────────────

print_step "Updating pyproject.toml to $VERSION_BARE..."

if [ "$DRY_RUN" = "true" ]; then
    print_info "Would update pyproject.toml version to $VERSION_BARE"
else
    sed -i '' "s/^version = \".*\"/version = \"$VERSION_BARE\"/" pyproject.toml
    PATH=/opt/homebrew/bin:$PATH git add pyproject.toml

    # Only commit if there are actually changes (version might already be correct)
    if PATH=/opt/homebrew/bin:$PATH git diff --cached --quiet; then
        print_info "Version already at $VERSION_BARE, no changes needed"
    else
        PATH=/opt/homebrew/bin:$PATH git commit -m "chore: bump version to $VERSION_BARE"
        print_success "pyproject.toml updated and committed"
    fi
fi

# ── Build release notes ───────────────────────────────────────────────────────

print_step "Building release notes..."

RELEASE_NOTES=""

# 1. Try to extract from CHANGELOG — look for ## [X.Y.Z] or ## [Unreleased]
if [ -f "$CHANGELOG" ]; then
    CHANGELOG_ENTRY=$(awk "/^## \[${VERSION_BARE}\]/,/^## \[/" "$CHANGELOG" \
        | sed '$d' | sed '1d' | sed '/^[[:space:]]*$/d' || true)

    if [ -z "$CHANGELOG_ENTRY" ]; then
        CHANGELOG_ENTRY=$(awk '/^## \[Unreleased\]/,/^## \[/' "$CHANGELOG" \
            | sed '$d' | sed '1d' | sed '/^[[:space:]]*$/d' || true)
    fi
    [ -n "$CHANGELOG_ENTRY" ] && RELEASE_NOTES="$CHANGELOG_ENTRY"
fi

# 2. Generate from commits if no CHANGELOG entry
if [ -z "$RELEASE_NOTES" ]; then
    print_info "No CHANGELOG entry found — generating from commits"

    # Collect full diffs for LLM context
    DIFF_STAT=$(PATH=/opt/homebrew/bin:$PATH git diff "${LAST_TAG}..HEAD" --stat)

    # Use meaningful commits only, excluding release-machinery
    SUBJECTS=$(PATH=/opt/homebrew/bin:$PATH git log "${LAST_TAG}..HEAD" \
        --format="%s" \
        | grep -vE "^chore: (release changes|pre-release changes|bump version to )")

    if [ -z "$SUBJECTS" ]; then
        SUBJECTS="Maintenance release"
    fi

    # Format one commit line: strip type prefix, capitalize, append scope in backticks
    fmt_line() {
        local raw="$1"
        local scope msg
        scope=$(echo "$raw" | sed -nE 's/^[a-z]+\(([^)]+)\)(!)?:.*/\1/p')
        msg=$(echo "$raw" | sed -E 's/^[a-z]+(\([^)]+\))?(!)?:[[:space:]]*//')
        msg=$(echo "$msg" | awk '{$1=toupper(substr($1,1,1))substr($1,2); print}')
        if [ -n "$scope" ]; then
            echo "- $msg (\`$scope\`)"
        else
            echo "- $msg"
        fi
    }

    build_section() {
        local header="$1" commits="$2" out=""
        [ -z "$commits" ] && return
        out="### $header"$'\n'
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            out+=$(fmt_line "$line")$'\n'
        done <<< "$commits"
        echo "$out"
    }

    FEATS=$(echo "$SUBJECTS"    | grep -E "^feat(\([^)]+\))?(!)?:"     || true)
    FIXES=$(echo "$SUBJECTS"    | grep -E "^fix(\([^)]+\))?:"          || true)
    PERF=$(echo "$SUBJECTS"     | grep -E "^perf(\([^)]+\))?:"         || true)
    REFACTOR=$(echo "$SUBJECTS" | grep -E "^refactor(\([^)]+\))?:"     || true)
    STYLE=$(echo "$SUBJECTS"    | grep -E "^style(\([^)]+\))?:"        || true)
    DOCS=$(echo "$SUBJECTS"     | grep -E "^docs(\([^)]+\))?:"         || true)
    CHORE=$(echo "$SUBJECTS"    | grep -E "^chore(\([^)]+\))?:"        || true)
    OTHER=$(echo "$SUBJECTS"    | grep -vE "^(feat|fix|perf|refactor|style|docs|test|chore|ci|build)(\([^)]+\))?(!)?:" || true)

    COMMIT_NOTES=""
    S=$(build_section "✨ New Features"   "$FEATS")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "🐛 Bug Fixes"      "$FIXES")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "⚡ Performance"    "$PERF")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "♻️ Improvements"   "$REFACTOR")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "💅 UI & Styling"   "$STYLE")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "📚 Documentation"  "$DOCS")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "🔧 Maintenance"    "$CHORE")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'
    S=$(build_section "📦 Other"          "$OTHER")
    [ -n "$S" ] && COMMIT_NOTES+="$S"$'\n'

    # Fallback if all sections ended up empty
    if [ -z "$COMMIT_NOTES" ]; then
        COMMIT_NOTES="### 📦 Changes"$'\n'
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            COMMIT_NOTES+="- $line"$'\n'
        done <<< "$SUBJECTS"
    fi

    # 3. Generate a human-readable summary using local Ollama
    OLLAMA_URL="http://localhost:11434/v1/chat/completions"
    OLLAMA_MODEL="qwen3.5:4b"
    HUMAN_SUMMARY=""
    if curl -sf "$OLLAMA_URL" -X POST -H "Content-Type: application/json" \
         -d '{"model":"'"$OLLAMA_MODEL"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
         &>/dev/null; then
        print_info "Generating release summary with Ollama ($OLLAMA_MODEL)..."

        # Build JSON payload safely using node to escape strings
        PAYLOAD=$(PATH=/opt/homebrew/bin:$PATH node -e "
            const prompt = 'You are writing a release summary for a cattle auction video pipeline CLI (Python, yt-dlp, Whisper, OCR, LLM extraction).\n\n' +
                'Version: $VERSION_TAG (previous: $LAST_TAG)\n\n' +
                'Commits:\n' + $(echo "$SUBJECTS" | PATH=/opt/homebrew/bin:$PATH node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>console.log(JSON.stringify(d)))") + '\n\n' +
                'Files changed:\n' + $(echo "$DIFF_STAT" | PATH=/opt/homebrew/bin:$PATH node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>console.log(JSON.stringify(d)))") + '\n\n' +
                'Write a concise 1-3 sentence summary of what changed in this release and why it matters to the user. Focus on the user-facing impact, not implementation details. Write in English. No markdown formatting, just plain text. No thinking tags.';
            console.log(JSON.stringify({
                model: '$OLLAMA_MODEL',
                messages: [{role:'user', content: prompt}],
                temperature: 0.3,
                max_tokens: 300
            }));
        ")

        RESPONSE=$(curl -sf "$OLLAMA_URL" -X POST \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" 2>/dev/null || true)

        if [ -n "$RESPONSE" ]; then
            SUMMARY_RESULT=$(echo "$RESPONSE" | PATH=/opt/homebrew/bin:$PATH node -e "
                let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{
                    try {
                        const r=JSON.parse(d);
                        let text = r.choices?.[0]?.message?.content ?? '';
                        // Strip <think>...</think> tags if present
                        text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                        process.stdout.write(text);
                    } catch { process.exit(1); }
                })
            " 2>/dev/null || true)
            if [ -n "$SUMMARY_RESULT" ]; then
                HUMAN_SUMMARY="$SUMMARY_RESULT"
                print_success "Summary generated"
            else
                print_info "Ollama returned empty response — skipping summary"
            fi
        else
            print_info "Ollama request failed — skipping summary"
        fi
    else
        print_info "Ollama not available at $OLLAMA_URL — skipping summary"
    fi

    if [ -n "$HUMAN_SUMMARY" ]; then
        RELEASE_NOTES="${HUMAN_SUMMARY}"$'\n\n'"${COMMIT_NOTES}"
    else
        RELEASE_NOTES="$COMMIT_NOTES"
    fi
fi

print_success "Release notes ready:"
echo "$RELEASE_NOTES" | head -20
[ "$(echo "$RELEASE_NOTES" | wc -l)" -gt 20 ] && print_dim "..."

# ── Update CHANGELOG.md ─────────────────────────────────────────────────────

print_step "Updating CHANGELOG.md..."

TODAY=$(date +%Y-%m-%d)
ENTRY="## [$VERSION_BARE] — $TODAY"$'\n\n'"$RELEASE_NOTES"

if [ "$DRY_RUN" = "true" ]; then
    print_info "Would prepend entry to CHANGELOG.md"
else
    if [ -f "$CHANGELOG" ]; then
        # Insert after the first line (# Changelog header) or at the top
        if head -1 "$CHANGELOG" | grep -q "^# "; then
            HEADER=$(head -1 "$CHANGELOG")
            BODY=$(tail -n +2 "$CHANGELOG")
            printf '%s\n\n%s\n%s\n' "$HEADER" "$ENTRY" "$BODY" > "$CHANGELOG"
        else
            printf '%s\n\n%s' "$ENTRY" "$(cat "$CHANGELOG")" > "$CHANGELOG"
        fi
    else
        printf '# Changelog\n\n%s\n' "$ENTRY" > "$CHANGELOG"
    fi
    PATH=/opt/homebrew/bin:$PATH git add "$CHANGELOG"
    if ! PATH=/opt/homebrew/bin:$PATH git diff --cached --quiet; then
        PATH=/opt/homebrew/bin:$PATH git commit -m "docs: update CHANGELOG for $VERSION_TAG"
        print_success "CHANGELOG.md updated"
    fi
fi

# ── Tag ───────────────────────────────────────────────────────────────────────

print_step "Creating tag $VERSION_TAG..."

if [ "$DRY_RUN" = "true" ]; then
    print_info "Would create: git tag -a $VERSION_TAG -m \"Release $VERSION_TAG\""
else
    PATH=/opt/homebrew/bin:$PATH git tag -a "$VERSION_TAG" -m "Release $VERSION_TAG"
    print_success "Tag $VERSION_TAG created"
fi

# ── Push ──────────────────────────────────────────────────────────────────────

print_step "Pushing to $REMOTE..."

if [ "$DRY_RUN" = "true" ]; then
    print_info "Would push branch $BRANCH and tag $VERSION_TAG"
else
    PATH=/opt/homebrew/bin:$PATH git push "$REMOTE" "$BRANCH"
    print_success "Pushed branch $BRANCH"
    PATH=/opt/homebrew/bin:$PATH git push "$REMOTE" "$VERSION_TAG"
    print_success "Pushed tag $VERSION_TAG"
fi

# ── GitHub release ────────────────────────────────────────────────────────────

if [ "$SKIP_GITHUB" = "true" ]; then
    print_step "Skipping GitHub release (--no-github)"
else
    print_step "Creating GitHub release $VERSION_TAG..."
    TEMP_NOTES=$(mktemp)
    echo "$RELEASE_NOTES" > "$TEMP_NOTES"

    if [ "$DRY_RUN" = "true" ]; then
        print_info "Would create GitHub release: $VERSION_TAG"
        rm -f "$TEMP_NOTES"
    else
        if $GH_CMD release create "$VERSION_TAG" \
            --repo "$REPO" \
            --title "$VERSION_TAG" \
            --notes-file "$TEMP_NOTES"; then
            rm -f "$TEMP_NOTES"
            print_success "GitHub release published"
        else
            rm -f "$TEMP_NOTES"
            print_error "Failed to create GitHub release"
            print_info "Create manually: https://github.com/$REPO/releases/new?tag=$VERSION_TAG"
            exit 1
        fi
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
print_success "Released $VERSION_TAG"
[ "$SKIP_GITHUB" != "true" ] && echo "  https://github.com/$REPO/releases/tag/$VERSION_TAG"
