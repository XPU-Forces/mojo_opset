#!/usr/bin/env bash
# Build one native provider for development, or one distribution for release.
set -euo pipefail

usage() {
    printf 'Usage: bash build.sh <provider> [--wheel]\n       bash build.sh --wheel\nWithout --wheel, compile and install one editable lib package.\nWith --wheel, build only the selected lib wheel, or the Python main wheel if no provider is given.\nNative builds require a configured Python/vendor toolchain and source dependencies.\n'
}
build_wheel=0
provider=""
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --wheel) build_wheel=1 ;;
        *)
            if [[ -n "$provider" || "$arg" == -* ]]; then usage >&2; exit 2; fi
            provider="$arg"
            ;;
    esac
done
if [[ -z "$provider" ]] && ((!build_wheel)); then usage >&2; exit 2; fi
cd "$(dirname "$0")"

# The publishing service uploads every artifact in dist/. Do not mix releases.
if ((build_wheel)); then
    shopt -s nullglob
    existing=(dist/*.whl dist/*.tar.gz)
    if ((${#existing[@]})); then
        printf 'dist/ already contains release artifacts; move them out before building.\n' >&2
        exit 1
    fi
fi

if [[ -n "$provider" ]]; then
    export MOJO_NATIVE_PROVIDER="$provider"
    python - <<'PY'
from pathlib import Path
from _packaging import native_provider

native_provider(Path.cwd())
PY
    bash native/build.sh "$provider"
    if ((build_wheel)); then
        python -m pip wheel ./native --no-build-isolation --no-deps -w dist
    else
        python -m pip install --no-build-isolation --no-deps -e ./native
    fi
else
    python -m pip wheel . --no-build-isolation --no-deps -w dist
fi
