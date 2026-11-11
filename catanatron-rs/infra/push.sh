#!/usr/bin/env bash
# Builds the actor image for linux/amd64 and linux/arm64 and pushes it (one
# multi-arch tag) to the stack's ECR repo. Cross-compiles in a build stage,
# so no emulation is needed.
#   infra/push.sh                                     # both platforms, push
#   PLATFORMS=linux/amd64 infra/push.sh --local       # one platform, load locally
set -euo pipefail
cd "$(dirname "$0")/.."
builder=${BUILDER:-kamal-local-docker-container}   # a docker-container builder (multi-platform)
platforms=${PLATFORMS:-linux/amd64,linux/arm64}
output() {
  aws cloudformation describe-stacks --stack-name CatanSelfPlay \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
if [[ "${1:-}" == "--local" ]]; then
  docker buildx build --builder "$builder" --platform "$platforms" -f Dockerfile.actor -t catan-actor:local --load .
  exit 0
fi
repo=$(output RepoUri)
aws ecr get-login-password | docker login --username AWS --password-stdin "${repo%%/*}"
docker buildx build --builder "$builder" --platform "$platforms" -f Dockerfile.actor -t "$repo:latest" --push .
