#!/bin/bash
set -u
cd /root/llm/test/v100-opt/llama.cpp || exit 9
LOG=/tmp/slotdbg-build.log
: > ${LOG}
echo "LIBLLAMA_MD5_BEFORE=$(md5sum /root/libdir-instr/libllama.so.0.4.1 | awk '{print $1}')" | tee -a ${LOG}
cmake --build build-instr --config Release -j128 >> ${LOG} 2>&1
RC=$?
echo "BUILD_RC=${RC}"
echo "GREP_ERROR_LINES=$(grep -c -e error ${LOG})"
echo "GREP_ERROR_COLON_LINES=$(grep -c -e error: ${LOG})"
echo "GREP_BUILT_OR_BUILDING=$(grep -c -e 'Built target' -e 'Building' ${LOG})"
echo "GREP_CTX_TU=$(grep -c -e 'llama-context.cpp.o' ${LOG})"
if [ "${RC}" != "0" ]; then
  echo "--- last 40 log lines ---"
  tail -40 ${LOG}
  echo "BUILD_ABORT"
  exit 1
fi
cp -a build-instr/bin/. /root/libdir-instr/
echo "CP_RC=$?"
ls -la /root/libdir-instr/libllama.so.0.4.1
echo "LIBLLAMA_MD5_AFTER=$(md5sum /root/libdir-instr/libllama.so.0.4.1 | awk '{print $1}')"
echo "SLOT_GRAPH_ENV_STRINGS=$(strings /root/libdir-instr/libllama.so.0.4.1 | grep -c LLAMA_GRAPH_SLOT_DEBUG)"
echo "SLOT_FMT_STRINGS=$(strings /root/libdir-instr/libllama.so.0.4.1 | grep -c -F '[SLOT] ctx=')"
echo "SLOT_FMT_RAW=$(strings /root/libdir-instr/libllama.so.0.4.1 | grep -F '[SLOT] ctx=')"
echo "BUILD_DONE"
