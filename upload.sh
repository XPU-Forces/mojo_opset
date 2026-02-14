#!/bin/bash
read -sp "Enter JWT Token: " JWT_TOKEN
echo

PACKAGE_NAME="byted-mojo-opset"

DIST_DIR="$(dirname "$0")/dist"

FILES=($(find "$DIST_DIR" -maxdepth 1 -type f \( -name "*.whl" -o -name "*.tar.gz" \)))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .whl or .tar.gz files found in $DIST_DIR"
    exit 1
fi

echo "Found files:"
for f in "${FILES[@]}"; do
    echo "  $(basename $f)"
done

curl --location 'https://bytedpypi.byted.org/' \
--header "X-Jwt-Token: ${JWT_TOKEN}" \
--form "pypi.package=${PACKAGE_NAME}" \
$(for i in "${!FILES[@]}"; do
    echo -n "--form asset${i}.filename=$(basename ${FILES[$i]}) "
    echo -n "--form asset${i}=@${FILES[$i]} "
done)