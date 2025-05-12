#!/bin/bash
# Minimal load-to-staging script (no retraining)

set -e

IMAGE_REPO="registry.kube-system.svc.cluster.local:5000/mlops-app"
IMAGE_TAG="staging-$(date +%Y%m%d%H%M%S)"
HELM_VALUES="k8s/staging/values.yaml"

echo "[1/4] Registering existing model..."
python3 train/register_model.py

echo "[2/4] Building Docker image..."
docker build -t ${IMAGE_REPO}:${IMAGE_TAG} -f inference/Dockerfile.staging inference/

echo "[3/4] Pushing Docker image..."
docker push ${IMAGE_REPO}:${IMAGE_TAG}

echo "[4/4] Updating Helm image tag..."
sed -i.bak "s|tag: .*|tag: ${IMAGE_TAG}|" ${HELM_VALUES}

echo "Done! Now commit and push to deploy:"
echo "git commit -am 'Deploy existing model ${IMAGE_TAG}' && git push"
