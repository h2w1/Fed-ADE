#!/usr/bin/env bash
set -euo pipefail

# This helper downloads only datasets that are not handled automatically by torchvision.
# CIFAR-10 and CIFAR-100 are downloaded by torchvision when the experiment runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT_DIR/experiments/covariate_shift/CIFAR"
mkdir -p "$ROOT_DIR/experiments/covariate_shift/CIFAR100"
mkdir -p "$ROOT_DIR/experiments/label_shift/TinyImageNet"

# CIFAR-10-C
if [ ! -d "$ROOT_DIR/experiments/covariate_shift/CIFAR/CIFAR-10-C" ]; then
  echo "Downloading CIFAR-10-C..."
  wget -O "$ROOT_DIR/experiments/covariate_shift/CIFAR/CIFAR-10-C.tar" "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
  tar -xf "$ROOT_DIR/experiments/covariate_shift/CIFAR/CIFAR-10-C.tar" -C "$ROOT_DIR/experiments/covariate_shift/CIFAR"
fi

# CIFAR-100-C
if [ ! -d "$ROOT_DIR/experiments/covariate_shift/CIFAR100/CIFAR-100-C" ]; then
  echo "Downloading CIFAR-100-C..."
  wget -O "$ROOT_DIR/experiments/covariate_shift/CIFAR100/CIFAR-100-C.tar" "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
  tar -xf "$ROOT_DIR/experiments/covariate_shift/CIFAR100/CIFAR-100-C.tar" -C "$ROOT_DIR/experiments/covariate_shift/CIFAR100"
fi

# Tiny ImageNet
if [ ! -d "$ROOT_DIR/experiments/label_shift/TinyImageNet/tiny-imagenet-200" ]; then
  echo "Downloading Tiny ImageNet..."
  wget -O "$ROOT_DIR/experiments/label_shift/TinyImageNet/tiny-imagenet-200.zip" "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  unzip -q "$ROOT_DIR/experiments/label_shift/TinyImageNet/tiny-imagenet-200.zip" -d "$ROOT_DIR/experiments/label_shift/TinyImageNet"
fi

echo "Done."
