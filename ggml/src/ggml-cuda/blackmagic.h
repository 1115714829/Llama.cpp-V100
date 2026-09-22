// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// Black Magic - experimental-path gate for the sm70-attn fork.
//
// One master switch for every experimental kernel path added on top of the
// stock sm70 fork. When the environment variable FISHLIKEXIE_BLACK_MAGIC is
// explicitly set to "1", the experimental paths are enabled; otherwise (unset,
// "0", or any other value) the stock sm70 paths are used unchanged.
//
// Design contract (per user directive 2026-09-15):
//   * Default OFF. The stock path must be byte-for-byte unaffected when the
//     variable is unset or not "1".
//   * Each experimental path (register-only FA, SPLITK GEMM, W4A4 prefill,
//     dual-pipeline GEMV) checks black_magic_on() at its dispatch hook and
//     falls through to the stock path when it is false.
//   * Read once per process (static const cache) - zero per-call overhead.
//
// This header is intentionally self-contained: no CUDA, no ggml includes, so
// it can be pulled into any .cu / .cpp translation unit cheaply.

#pragma once

#include <cstdlib> // getenv

// Returns true iff FISHLIKEXIE_BLACK_MAGIC is set to a string whose first
// character is '1' (i.e. "1", "1...", etc.). Any other value - including
// unset, "0", "true", "yes" - returns false. Deliberately strict: this is a
// research gate, not a user-facing convenience flag, so only an explicit "1"
// opts in.
inline bool black_magic_on() {
    static const bool enabled = [] {
        const char * e = std::getenv("FISHLIKEXIE_BLACK_MAGIC");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}
