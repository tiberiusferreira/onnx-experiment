#!/bin/zsh
protoc --decode=onnx.ModelProto onnx_protobuf/onnx.proto3 < $1 > $1.txt