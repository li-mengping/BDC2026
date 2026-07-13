#!/usr/bin/env bash
set -euo pipefail

# 生成比赛提交用的离线 bundle。所有路径均从脚本位置推导，不依赖调用者的当前目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="bdc2026:latest"
PYTHON_IMAGE="python:3.10-slim-bookworm@sha256:ff7161e2b8e2a56fc6a62a6099ff8feb72f1a6dbae9860cdcb9a6c65cf4c6be9"
OUTPUT_NAME="${1:-bdc2026-bundle.tar}"
OUTPUT_NAME="$(basename -- "${OUTPUT_NAME}")"
[[ "${OUTPUT_NAME}" == *.tar ]] || OUTPUT_NAME="${OUTPUT_NAME}.tar"

DOCKERFILE_ARG="ARG PYTHON_IMAGE=${PYTHON_IMAGE}"
if ! grep -Fqx "${DOCKERFILE_ARG}" "${ROOT}/Dockerfile"; then
    echo "Dockerfile 未固定到约定的 Python 3.10 digest。" >&2
    exit 1
fi

DIST="${ROOT}/dist"
STAGING="${DIST}/bdc2026-bundle"
rm -rf "${STAGING}"
mkdir -p "${STAGING}/data" "${STAGING}/output" "${STAGING}/temp"

# 不拉取、不回退到代理镜像；缺失固定基础镜像时让 docker 直接报告错误。
docker build --pull=false --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE}" -t "${IMAGE}" "${ROOT}"
docker save -o "${STAGING}/bdc2026-image.tar" "${IMAGE}"

cp "${ROOT}/docker-compose.yml" "${STAGING}/docker-compose.yml"
cp "${ROOT}/readme.md" "${STAGING}/readme.md"
cp "${ROOT}/data/stock_data.csv" "${STAGING}/data/stock_data.csv"
cp "${ROOT}/data/manifest.json" "${STAGING}/data/manifest.json"
cp "${ROOT}/data/hs300_stock_list.csv" "${STAGING}/data/hs300_stock_list.csv"

(
    cd "${STAGING}"
    : > SHA256SUMS
    while IFS= read -r -d '' file; do
        sha256sum "${file#./}"
    done < <(find . -type f ! -name SHA256SUMS -print0 | sort -z) > SHA256SUMS
)

tar -cf "${DIST}/${OUTPUT_NAME}" -C "${STAGING}" .
echo "已生成离线发布 bundle: ${DIST}/${OUTPUT_NAME}"
echo "镜像: ${STAGING}/bdc2026-image.tar"
