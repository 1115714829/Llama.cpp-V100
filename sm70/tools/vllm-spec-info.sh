#!/bin/bash
# vllm-spec-info.sh - read-only: speculative-decoding settings from the vllm-1cat journal, secrets redacted.
journalctl -u vllm-1cat --no-pager -n 5000 2> /dev/null \
  | grep -i -E 'speculative|num_speculative|dflash|draft' \
  | sed -E "s/(api[_-]?key['\"]?[:= ]+['\"]?)[^'\", )]+/\1<REDACTED>/Ig" \
  | cut -c1-400 | tail -30
echo VLLM_SPEC_INFO_DONE
