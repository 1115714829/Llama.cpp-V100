#!/bin/bash
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr
LIB=/root/m5-lib
OBJ=/root/m5-obj
CXXBIN=/opt/rh/gcc-toolset-12/root/usr/bin/g++
DEFS="-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DLLAMA_SHARED -DLLAMA_SUBPROCESS"
INC="-I$TREE/src/../include -I$TREE/ggml/src/../include -I$TREE/common/. -I$TREE/vendor/nlohmann/.. -I$TREE/vendor/sheredom/.."
echo "lib_md5=$(md5sum $LIB/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
$CXXBIN $DEFS $INC -O3 -DNDEBUG -pthread -c /root/m5-src/test-backend-ops.cpp -o $OBJ/test-backend-ops.cpp.o > /root/m5-log/verify.cc.log 2>&1 || { echo VERIFY_CC_FAIL; tail -10 /root/m5-log/verify.cc.log; exit 1; }
$CXXBIN -O3 -DNDEBUG $OBJ/test-backend-ops.cpp.o -o $LIB/test-backend-ops -Wl,-rpath,$LIB $LIB/libllama-common.so.0.4.1 $LIB/libllama.so.0.4.1 $LIB/libggml.so.0.24.0 $LIB/libggml-cpu.so.0.24.0 $LIB/libggml-cuda.so.0.24.0 $LIB/libggml-base.so.0.24.0 $BD/common/libllama-common-base.a -lpthread > /root/m5-log/verify.ld.log 2>&1 || { echo VERIFY_LD_FAIL; tail -10 /root/m5-log/verify.ld.log; exit 1; }
echo VERIFY_BIN_BUILT
LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\]" 2>&1 | grep -a -e OK -e FAIL -e Error -e Backend | tail -30
echo M5_VERIFY_DONE
