#!/bin/bash
SRC=/root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh
cur=$(md5sum $SRC | cut -d' ' -f1)
echo "current_md5=$cur"
case "$cur" in
  4b17ca95e29496bf87d4b22ccbb4c3dd|98a6ed3eeb347aa774d74801ab55c9bd)
    cp /root/m5-pristine.cuh $SRC
    echo "restored_md5=$(md5sum $SRC | cut -d' ' -f1)"
    ;;
  *)
    echo "SKIP_UNEXPECTED_CONTENT"
    ;;
esac
