# models/ — 本机参考模型

- `vocab/`：从 `llama.cpp/models` 与 `v100-refs/**/models` 收拢的 **ggml-vocab-\*.gguf**（词表/测试用参考模型，去重后 19 个）。
- 大模型权重（Qwen3.8-27B Q8_0 / Q4_K_M / DFlash2 draft）**不在本机**，在服务器 `/mnt/3.84t/**`（见 AGENTS 环境边界）。
- 哈希与来源见 `olddoc/MANIFEST.md`。
- 研究主攻格式：**Q8_0**（见 `docs/v100-dev/05-量化范围.md`）。
