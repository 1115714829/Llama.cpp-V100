# v100-refs — 外部参考（只读）

| 目录 | 用途 |
|---|---|
| `vllm/` | 官方 vLLM |
| `1cat-vllm/` | 1cat 分支（抄作业对象） |
| **`vllm-1ca-vllm分支版本差异文件/`** | **分叉点基线**（原 `vllm-forkpoint`）= 1cat 起分支时的 vllm 版本；**从此处对照可看 1cat 的代码修改差异** |
| `flash-attention-v100/` `sm70-attn/` `v100-skinny/` `ninfer-v100/` `jusko-llama-volta-qwen3flash/` `sglang-V100/` `xllama.cpp/` `qwen38-v100-serve/` | V100 内核 / 服务参考实现 |

**看 1cat 相对分叉点的差异：**

```bash
git -C 1cat-vllm diff 4ff865c38..HEAD
git -C 1cat-vllm diff 4ff865c38..HEAD -- <path>
```

分叉点 commit 在 `1cat-vllm` 对象库内；`vllm-1ca-vllm分支版本差异文件/` 供肉眼对照。  
细节：[`../1cat-vllm-v100-study/vllm-vs-1cat-vllm-diff.md`](../1cat-vllm-v100-study/vllm-vs-1cat-vllm-diff.md)。

**禁止修改本树内任何内容。**
