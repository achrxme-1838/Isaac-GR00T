#!/usr/bin/env bash
# Host-side helper for the ~/gr00t_ws bind-mounted development container.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

usage() {
    cat <<'EOF'
Usage: bash docker/workspace.sh init | <docker compose arguments...>

Place this checkout at ~/gr00t_ws/Isaac-GR00T, then run:
  bash docker/workspace.sh init
  bash docker/workspace.sh build
  bash docker/workspace.sh up -d
  bash docker/workspace.sh exec gr00t bash
  bash docker/workspace.sh down

Optional environment variables:
  GR00T_WS            Host workspace (default: $HOME/gr00t_ws)
  GR00T_IMAGE         Image tag (default: gr00t:workspace)
  GR00T_PROJECT_NAME  Compose project (default: gr00t)
  GR00T_BIND_IP       Published inference address (default: 127.0.0.1)
  GR00T_PORT          Published inference port (default: 5555)

All Compose arguments are forwarded unchanged. Run this helper as your normal
host user so mounted files retain your UID/GID. See docker/SERVER_SETUP.ko.md.
EOF
}

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

export GR00T_WS="${GR00T_WS:-$HOME/gr00t_ws}"
if [[ "$GR00T_WS" != /* ]]; then
    echo "GR00T_WS must be an absolute path (for example: $HOME/gr00t_ws)." >&2
    exit 1
fi

# Ensure the checkout used to build the image is also the one used at runtime.
if [[ ! -d "$GR00T_WS/Isaac-GR00T" ]] || \
    [[ "$(cd -- "$GR00T_WS/Isaac-GR00T" && pwd -P)" != "$REPO_ROOT" ]]; then
    echo "Place this checkout at $GR00T_WS/Isaac-GR00T before running this helper." >&2
    echo "See $SCRIPT_DIR/SERVER_SETUP.ko.md for transfer commands." >&2
    exit 1
fi
GR00T_WS="$(cd -- "$GR00T_WS" && pwd -P)"
GR00T_HOST_UID="$(id -u)"
GR00T_HOST_GID="$(id -g)"
export GR00T_HOST_UID GR00T_HOST_GID

if [[ "$1" == "init" ]]; then
    mkdir -p -- "$GR00T_WS"/{models,datasets,outputs,logs,home,cache}
    mkdir -p -- "$GR00T_WS/cache"/{uv,huggingface,torch,torch_extensions,triton,cuda}
    echo "Workspace ready: $GR00T_WS -> /workspace/gr00t_ws"
    echo "Container UID:GID: $GR00T_HOST_UID:$GR00T_HOST_GID"
    exit 0
fi

for directory in models datasets outputs logs home cache; do
    if [[ ! -d "$GR00T_WS/$directory" || ! -w "$GR00T_WS/$directory" ]]; then
        echo "Missing or unwritable directory: $GR00T_WS/$directory" >&2
        echo 'Run "bash docker/workspace.sh init" as the workspace owner first.' >&2
        exit 1
    fi
done

exec docker compose \
    --project-name "${GR00T_PROJECT_NAME:-gr00t}" \
    --env-file /dev/null \
    --file "$SCRIPT_DIR/compose.yaml" \
    "$@"
