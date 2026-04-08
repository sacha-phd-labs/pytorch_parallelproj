#!/bin/sh
set -e

# Clone repositories for editable installs
REPOS_DIR="/workspace"
OWNER="sacha-phd-labs"
mkdir -p "$REPOS_DIR"

# Define repository URLs
declare -A REPOS=(
    ["pet-phantom"]="https://github.com/${OWNER}/pet-phantom.git"
    ["pytorch-utilities"]="https://github.com/${OWNER}/pytorch-utilities.git"
    ["toolbox"]="https://github.com/${OWNER}/toolbox.git"
    ["pet-simulation"]="https://github.com/${OWNER}/pet-simulation.git"
)

# Clone each repository
for repo_name in "${!REPOS[@]}"; do
    repo_url="${REPOS[$repo_name]}"
    echo "Cloning $repo_name from $repo_url..."
    git clone "$repo_url" "$REPOS_DIR/$repo_name"
done

# Install editable packages
pip install -e "$REPOS_DIR/pet-phantom" \
            -e "$REPOS_DIR/pytorch-utilities" \
            -e "$REPOS_DIR/toolbox" \
            -e "$REPOS_DIR/pet-simulation" \
            --no-deps

exec "$@"