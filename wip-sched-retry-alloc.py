#!/usr/bin/env python3
"""Retry the gallocr plan reuse before paying a device sync + a full re-reserve.

R248-R253 measured that ggml_backend_sched_alloc_splits takes the slow path on ~every call
(560 per arm) and that 99.3% of those are `backend_ids_changed` alone, not a real allocation
failure: a decode round alternates between several graphs (target verify / draft inject / draft
block / KV copy / prompt chunks), so the per-node backend id vector differs from the previous
call while the recorded buffer assignments are still valid. The slow path synchronizes all three
devices and re-runs the whole reserve, which is what leaves the GPU idle waiting for the host.

This adds an env-gated retry ([GGML_SCHED_RETRY_ALLOC]) that calls ggml_gallocr_alloc_graph()
once before falling back to the existing path. Default off = today's behavior byte for byte.

Usage: wip-sched-retry-alloc.py <path/to/ggml-backend.cpp> <apply|restore>
"""
import hashlib
import os
import sys

APPLY = [
    (
        "    int debug_realloc;\r\n"
        "    int debug_graph_size;\r\n"
        "    int debug_prev_graph_size;\r\n",
        "    int debug_realloc;\r\n"
        "    int debug_graph_size;\r\n"
        "    int debug_prev_graph_size;\r\n"
        "\r\n"
        "    // retry buffer plan reuse before a device sync + full reserve [GGML_SCHED_RETRY_ALLOC]\r\n"
        "    int retry_alloc;\r\n"
        "    long long n_bic;\r\n"
        "    long long n_retry;\r\n"
        "    long long n_retry_ok;\r\n"
        "    long long n_retry_bad;\r\n",
    ),
    (
        "    const char * GGML_SCHED_DEBUG_REALLOC = getenv(\"GGML_SCHED_DEBUG_REALLOC\");\r\n"
        "    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;\r\n",
        "    const char * GGML_SCHED_DEBUG_REALLOC = getenv(\"GGML_SCHED_DEBUG_REALLOC\");\r\n"
        "    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;\r\n"
        "\r\n"
        "    sched->retry_alloc = getenv(\"GGML_SCHED_RETRY_ALLOC\") != nullptr;\r\n"
        "    sched->n_bic       = 0;\r\n"
        "    sched->n_retry     = 0;\r\n"
        "    sched->n_retry_ok  = 0;\r\n"
        "    sched->n_retry_bad = 0;\r\n",
    ),
    (
        "    // allocate graph\r\n"
        "    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {\r\n",
        "    // A round alternates between graphs, so the backend id vector changes on almost every call\r\n"
        "    // while the recorded assignments can still apply. Try the plan first, and only then pay for\r\n"
        "    // a device sync and a full reserve. [GGML_SCHED_RETRY_ALLOC]\r\n"
        "    bool alloc_ok = false;\r\n"
        "    if (!backend_ids_changed) {\r\n"
        "        alloc_ok = ggml_gallocr_alloc_graph(sched->galloc, &sched->graph);\r\n"
        "    } else {\r\n"
        "        sched->n_bic++;\r\n"
        "        if (sched->retry_alloc) {\r\n"
        "            sched->n_retry++;\r\n"
        "            alloc_ok = ggml_gallocr_alloc_graph(sched->galloc, &sched->graph);\r\n"
        "            if (alloc_ok) {\r\n"
        "                // the reused plan is only valid if every node still sits in its backend's buffer\r\n"
        "                for (int i = 0; i < sched->graph.n_nodes; i++) {\r\n"
        "                    const struct ggml_tensor * node = sched->graph.nodes[i];\r\n"
        "                    if (node->view_src != NULL || node->buffer == NULL) {\r\n"
        "                        continue;\r\n"
        "                    }\r\n"
        "                    if (ggml_backend_buffer_get_type(node->buffer) != sched->bufts[sched->node_backend_ids[i]]) {\r\n"
        "                        alloc_ok = false;\r\n"
        "                        sched->n_retry_bad++;\r\n"
        "                        break;\r\n"
        "                    }\r\n"
        "                }\r\n"
        "            }\r\n"
        "            if (alloc_ok) {\r\n"
        "                sched->n_retry_ok++;\r\n"
        "            }\r\n"
        "        }\r\n"
        "    }\r\n"
        "\r\n"
        "    // allocate graph\r\n"
        "    if (!alloc_ok) {\r\n",
    ),
    (
        "void ggml_backend_sched_free(ggml_backend_sched_t sched) {\r\n"
        "    if (sched == NULL) {\r\n"
        "        return;\r\n"
        "    }\r\n",
        "void ggml_backend_sched_free(ggml_backend_sched_t sched) {\r\n"
        "    if (sched == NULL) {\r\n"
        "        return;\r\n"
        "    }\r\n"
        "    if (sched->retry_alloc) {\r\n"
        "        fprintf(stderr, \"[SCHED_RETRY] bic=%lld retry=%lld ok=%lld bad=%lld\\n\",\r\n"
        "                sched->n_bic, sched->n_retry, sched->n_retry_ok, sched->n_retry_bad);\r\n"
        "    }\r\n",
    ),
]


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "apply"
    orig = path + ".orig-schedretry"

    if mode == "restore":
        if not os.path.exists(orig):
            sys.exit("FAIL no pristine copy at " + orig)
        with open(orig, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data)
        print("RESTORED md5=" + md5_of(path))
        return

    with open(path, "rb") as f:
        text = f.read().decode("utf-8")

    if "GGML_SCHED_RETRY_ALLOC" in text:
        print("ALREADY_PATCHED md5=" + md5_of(path))
        return

    if not os.path.exists(orig):
        with open(orig, "wb") as f:
            f.write(text.encode("utf-8"))

    for i, (old, new) in enumerate(APPLY):
        n = text.count(old)
        if n != 1:
            sys.exit("FAIL anchor %d count = %d" % (i, n))
        text = text.replace(old, new, 1)

    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print("PATCHED md5=" + md5_of(path))


if __name__ == "__main__":
    main()
