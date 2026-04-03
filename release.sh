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
        print_info "Would commit all changes (git add -A)"
    else
        PATH=/opt/homebrew/bin:$PATH git add -A
        PATH=/opt/homebrew/bin:$PATH git commit -m "chore: release changes"
        print_success "Changes committed"
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

# Commits since last tag
COMMITS=$(PATH=/opt/homebrew/bin:$PATH git log "${LAST_TAG}..HEAD" --oneline)
COMMIT_COUNT=$(echo "$COMMITS" | grep -c . || true)

if [ "$COMMIT_COUNT" -eq 0 ]; then
    print_error "No commits since $LAST_TAG — nothing to release"
    exit 1
fi

print_info "Commits since $LAST_TAG: $COMMIT_COUNT"
echo "$COMMITS" | while read -r line; do print_dim "$line"; done

if [ -z "$VERSION" ]; then
    # feat: (new feature) → bump minor, reset patch
    # fix:/chore:/docs:/refactor:/etc → bump patch only
    if echo "$COMMITS" | grep -qE "^[a-f0-9]+ feat(\([^)]+\))?(!)?:"; then
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

# ── Update pyproject.toml ─────────────────────────────────────────────────────

print_step "Updating pyproject.toml to $VERSION_BARE..."

if [ "$DRY_RUN" = "true" ]; then
    print_info "Would update pyproject.toml version to $VERSION_BARE"
else
    sed -i '' "s/^version = \".*\"/version = \"$VERSION_BARE\"/" pyproject.toml
    PATH=/opt/homebrew/bin:$PATH git add pyproject.toml
    PATH=/opt/homebrew/bin:$PATH git commit -m "chore: bump version to $VERSION_BARE"
    print_success "pyproject.toml updated and committed"
fi

# ── Build release notes ───────────────────────────────────────────────────────

print_step "Building release notes..."

RELEASE_NOTES=""

# 1. Try to extract from CHANGELOG — look for ## [X.Y.Z] or ## [Unreleased]
if [ -f "$CHANGELOG" ]; then
    # Try exact version match first
    RELEASE_NOTES=$(awk "/^## \[${VERSION_BARE}\]/,/^## \[/" "$CHANGELOG" \
        | sed '$d' | sed '1d' | sed '/^[[:space:]]*$/d' || true)

    # Fall back to [Unreleased] section
    if [ -z "$RELEASE_NOTES" ]; then
        RELEASE_NOTES=$(awk '/^## \[Unreleased\]/,/^## \[/' "$CHANGELOG" \
            | sed '$d' | sed '1d' | sed '/^[[:space:]]*$/d' || true)
    fi
fi

# 2. Fall back to git log grouped by type
if [ -z "$RELEASE_NOTES" ]; then
    print_info "No CHANGELOG entry found — generating from commits"
    FEATURES=$(echo "$COMMITS" | grep -E "^[a-f0-9]+ feat(\([^)]+\))?(!)?:" | sed 's/^[a-f0-9]* /- /' || true)
    FIXES=$(echo "$COMMITS"    | grep -E "^[a-f0-9]+ fix(\([^)]+\))?:"  | sed 's/^[a-f0-9]* /- /' || true)
    OTHER=$(echo "$COMMITS"    | grep -vE "^[a-f0-9]+ (feat|fix)(\([^)]+\))?(!)?:" | sed 's/^[a-f0-9]* /- /' || true)

    [ -n "$FEATURES" ] && RELEASE_NOTES="${RELEASE_NOTES}### Added"$'\n'"${FEATURES}"$'\n\n'
    [ -n "$FIXES" ]    && RELEASE_NOTES="${RELEASE_NOTES}### Fixed"$'\n'"${FIXES}"$'\n\n'
    [ -n "$OTHER" ]    && RELEASE_NOTES="${RELEASE_NOTES}### Other"$'\n'"${OTHER}"$'\n'
fi

print_success "Release notes ready:"
echo "$RELEASE_NOTES" | head -10
[ "$(echo "$RELEASE_NOTES" | wc -l)" -gt 10 ] && print_dim "..."

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
