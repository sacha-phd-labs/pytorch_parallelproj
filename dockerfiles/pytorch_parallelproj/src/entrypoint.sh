#!/bin/bash
set -e

# Start SSH server
/usr/sbin/sshd -D &

# Clone repositories for editable installs
REPOS_DIR="/workspace"
OWNER="sacha-phd-labs"
mkdir -p "$REPOS_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate parallelproj_env

# Install only if not already installed (important!)
if ! python -c "import parallelproj" &> /dev/null; then
    conda install -y -c conda-forge parallelproj pytorch cupy
    pip install --no-cache-dir -r /tmp/requirements.txt
    conda clean -afy
fi

# Define repository URLs
declare -A REPOS=(
    ["pet-phantom"]="https://github.com/${OWNER}/pet-phantom.git"
    ["pytorch-utilities"]="https://github.com/${OWNER}/pytorch-utilities.git"
    ["toolbox"]="https://github.com/${OWNER}/toolbox.git"
    ["pet-simulation"]="https://github.com/${OWNER}/pet-simulation.git"
    ["noise2noise"]="https://github.com/${OWNER}/noise2noise.git"
)

# Clone each repository
for repo_name in "${!REPOS[@]}"; do
    if [ -d "$REPOS_DIR/$repo_name/.git" ]; then
        echo "Repository $repo_name already exists in $REPOS_DIR. Skipping clone."
        continue
    fi
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