#!/usr/bin/env bash
set -euo pipefail

# 手动最终打包脚本：构建 bdc2026 镜像，并导出未压缩 Docker tar。
# 用法：bash package.sh 队伍名称.tar
output_tar="${1:-HFUT_001.tar}"
image_name="bdc2026"
python_image="${PYTHON_IMAGE:-python:3.12-slim-bookworm}"

if ! docker image inspect "${python_image}" >/dev/null 2>&1; then
    candidates=(
        "${python_image}"
        "docker.m.daocloud.io/library/python:3.12-slim-bookworm"
        "dockerproxy.com/library/python:3.12-slim-bookworm"
        "docker.1ms.run/library/python:3.12-slim-bookworm"
        "dockerpull.com/library/python:3.12-slim-bookworm"
    )

    pulled_image=""
    for candidate in "${candidates[@]}"; do
        echo "Trying base image: ${candidate}"
        if docker pull --platform linux/amd64 "${candidate}"; then
            pulled_image="${candidate}"
            break
        fi
    done

    if [[ -z "${pulled_image}" ]]; then
        echo "Failed to pull python:3.12-slim-bookworm from all configured sources." >&2
        echo "Set PYTHON_IMAGE to a reachable local or remote image and rerun." >&2
        exit 1
    fi

    docker tag "${pulled_image}" "${python_image}"
fi

docker buildx build --platform linux/amd64 --build-arg PYTHON_IMAGE="${python_image}" -t "${image_name}:latest" .
docker save -o "${output_tar}" "${image_name}:latest"

echo "Docker image saved to ${output_tar}"
