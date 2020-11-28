#!/bin/zsh
(cd "$(dirname "$0")" && source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate && python3 ./onnx_train.py "$@")