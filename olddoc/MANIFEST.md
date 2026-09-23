# olddoc / models 回档清单

生成时间: 2026-09-23T11:34:47.2221801+08:00
规则: 先复制到目标并校验 SHA256，再从源删除（等同移动，源可按本清单+git 恢复）

## 警告
- qwen-private/QWEN.md **含服务器口令等机密**，禁止提交/分享/复制进公开仓库。
- 工作区未发现独立的 DeepSeek 私有文件夹；若你另有目录请指出路径再迁。

## 一、Qwen / 旧私有 -> olddoc/qwen-private
- FILE F:\vllm+llama.cpp\QWEN.md -> F:\vllm+llama.cpp\olddoc\qwen-private\QWEN.md  sha256=FE7201C7DFD8CD592F5273E9D1A1791F615F028BE22AF36AD51BDC599C380633  verify=OK  bytes=62840
- DIR  F:\vllm+llama.cpp\.qwen -> F:\vllm+llama.cpp\olddoc\qwen-private\.qwen  files 27 -> 27  OK

## 二、根目录散落旧件 -> olddoc/workspace-loose
- FILE AGENTS.md  sha256=5847F751DEE779655FD2996DDD4BA66DF8598A96F03C6305618649D3A74208BB  verify=OK  bytes=65475
- FILE ARCHIVE-COMMIT-MSG.txt  sha256=39C1E05BDBD40150C2B5F5945A8B28BBD8C357FA622D3442168A252842A19129  verify=OK  bytes=1500
- FILE local-lf-ggml-cuda.cu  sha256=CB2F139D8E370B885592D0B82CE9A897E2C0255CC442597656444CA3E5CB5C55  verify=OK  bytes=261224
- FILE .sshprobe.txt  sha256=E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855  verify=OK  bytes=0
- FILE .sshprobe.txt.err  sha256=C20C03F43C6CAF739E8CBBF91721BA8257B8E5EBFE3DDB6019A6E7F63AEBBDAC  verify=OK  bytes=39
- FILE             3633 File(s)    633,083,426 bytes  sha256=5FED9A539DCD48644E160A4EA1CFDFF360500C4D7938A4607EDE86C869D5907C  verify=OK  bytes=13
- FILE             6445 File(s)    382,135,413 bytes  sha256=5BCD9F5D143ECAA0AE8C77294192117262A44697B0F3C675DA7A9A2B1F063B8F  verify=OK  bytes=13
- FILE             7245 File(s)    432,825,299 bytes  sha256=00727ECE160E3405CB28F414CBB196804B0F8F14CAF5F0A3E4576065D4A28757  verify=OK  bytes=8

## 三、.tmpcmp 工件 -> olddoc/tmp-artifacts
- DIR  .tmpcmp -> olddoc/tmp-artifacts/.tmpcmp  files=2

## 四、本机参考模型 GGUF -> models/vocab

## 五、校验摘要
- 目标目录 olddoc 与 models 已写入；源文件在校验 OK 后删除（下一步）。
- AGENTS.md 仍需在工作区根保留一份有效副本供自动加载：迁移后从 olddoc/workspace-loose/AGENTS.md **复制**回根（不是只留旧档）。

## 四（补）本机参考模型 GGUF -> models/vocab  2026-09-23T11:35:22.1697832+08:00
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-aquila.gguf -> models/vocab/ggml-vocab-aquila.gguf  sha256=7C53C3C516AC67C7CA12977B9690FDEA3D2EF13BBAED6378F98191A13EF5CA00  verify=OK  bytes=4825676
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-baichuan.gguf -> models/vocab/ggml-vocab-baichuan.gguf  sha256=4F5B955697F3BD3108070B1D5936C7EB9FC542B81C6932E59ABDDEC75BCA1963  verify=OK  bytes=1340998
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-bert-bge.gguf -> models/vocab/ggml-vocab-bert-bge.gguf  sha256=FBCBE22278FB302694D5F4A41BFE48C5F90E8E3554EAB1C0435387DFF654A854  verify=OK  bytes=627549
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-command-r.gguf -> models/vocab/ggml-vocab-command-r.gguf  sha256=A2F8CFEA952EF7C391A6D92A1C309D0BD32E36384D9B9230569A7425732F27D9  verify=OK  bytes=10874545
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-deepseek-coder.gguf -> models/vocab/ggml-vocab-deepseek-coder.gguf  sha256=91CB1379F2E33AF1C4866B194622B7A0E12E8F0C9DBA7BA2F10D55978730BEC1  verify=OK  bytes=1156067
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-deepseek-llm.gguf -> models/vocab/ggml-vocab-deepseek-llm.gguf  sha256=867F77537B54565F0D81D508C04EDC41AA1D4FFC1A92745F225B4C1B02755F76  verify=OK  bytes=3970167
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-falcon.gguf -> models/vocab/ggml-vocab-falcon.gguf  sha256=9F0BF8B0733680398B72E652E90F260F43782F326E75545FC0E49611A5BA35AD  verify=OK  bytes=2287728
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-gemma-4.gguf -> models/vocab/ggml-vocab-gemma-4.gguf  sha256=58B1BA0B57F3B4D7C468BA4FFD91AD85190346A3D7AD7E71D1CABAAE8A14BB65  verify=OK  bytes=15776467
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-gpt-2.gguf -> models/vocab/ggml-vocab-gpt-2.gguf  sha256=CEDC56CA6E2E89F63E781696D1FD76B4B1D49E6720DEE86463E915F6E90016AC  verify=OK  bytes=1766807
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-gpt-neox.gguf -> models/vocab/ggml-vocab-gpt-neox.gguf  sha256=AE593A7F9B8BB174ED4F5019E41530463E4DAC7AA06E42DEE8AA650D2BDAC53D  verify=OK  bytes=1771431
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-llama-bpe.gguf -> models/vocab/ggml-vocab-llama-bpe.gguf  sha256=97272E430D53BC7688F52D5E0AD8EA8F163EDE9F1BBD1694FEAA504797D5D96E  verify=OK  bytes=7818140
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-llama-spm.gguf -> models/vocab/ggml-vocab-llama-spm.gguf  sha256=16C3724582D59AA8BF84711894E833F916EE46A31D80E21312759C48BF8D0E69  verify=OK  bytes=723869
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-mpt.gguf -> models/vocab/ggml-vocab-mpt.gguf  sha256=59DC382612866D1FC6C11EA531318D327598F3412D9C8F8600607CDF3030898F  verify=OK  bytes=1771393
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-nomic-bert-moe.gguf -> models/vocab/ggml-vocab-nomic-bert-moe.gguf  sha256=90A6746926454784A98389AD36A36D89BC9CFC81DB9CB0F33C941BCC959FE5F9  verify=OK  bytes=6821877
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-phi-3.gguf -> models/vocab/ggml-vocab-phi-3.gguf  sha256=967D7190D11C4842EAB697079D98D56C2116E10EB617BE355A2733BFC132E326  verify=OK  bytes=726019
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-qwen2.gguf -> models/vocab/ggml-vocab-qwen2.gguf  sha256=44C2F46B715F585C6AB513970E8A006BFA5BADD6108560054921CF598D154D8C  verify=OK  bytes=5928681
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-qwen35.gguf -> models/vocab/ggml-vocab-qwen35.gguf  sha256=63ED952FF338996CF0BDF24A7B10015124273F75C6DC9BB427356AA3F67EC62C  verify=OK  bytes=5928682
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-refact.gguf -> models/vocab/ggml-vocab-refact.gguf  sha256=AC3CEDA902FED91CCF74312B305D9B86C37E4F8E35FA9CC6EF3CE34FCA7D4678  verify=OK  bytes=1720710
- MODEL F:\vllm+llama.cpp\llama.cpp\models\ggml-vocab-starcoder.gguf -> models/vocab/ggml-vocab-starcoder.gguf  sha256=FEDB892B4E1BD3C1F2FCDAE356440B14FB458F4264D586E5C987ED93DF4E174D  verify=OK  bytes=1719346
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-aquila.gguf  sha256=7C53C3C516AC67C7CA12977B9690FDEA3D2EF13BBAED6378F98191A13EF5CA00  (= ggml-vocab-aquila.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-baichuan.gguf  sha256=4F5B955697F3BD3108070B1D5936C7EB9FC542B81C6932E59ABDDEC75BCA1963  (= ggml-vocab-baichuan.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-bert-bge.gguf  sha256=FBCBE22278FB302694D5F4A41BFE48C5F90E8E3554EAB1C0435387DFF654A854  (= ggml-vocab-bert-bge.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-command-r.gguf  sha256=A2F8CFEA952EF7C391A6D92A1C309D0BD32E36384D9B9230569A7425732F27D9  (= ggml-vocab-command-r.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-deepseek-coder.gguf  sha256=91CB1379F2E33AF1C4866B194622B7A0E12E8F0C9DBA7BA2F10D55978730BEC1  (= ggml-vocab-deepseek-coder.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-deepseek-llm.gguf  sha256=867F77537B54565F0D81D508C04EDC41AA1D4FFC1A92745F225B4C1B02755F76  (= ggml-vocab-deepseek-llm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-falcon.gguf  sha256=9F0BF8B0733680398B72E652E90F260F43782F326E75545FC0E49611A5BA35AD  (= ggml-vocab-falcon.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-gemma-4.gguf  sha256=58B1BA0B57F3B4D7C468BA4FFD91AD85190346A3D7AD7E71D1CABAAE8A14BB65  (= ggml-vocab-gemma-4.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-gpt-2.gguf  sha256=CEDC56CA6E2E89F63E781696D1FD76B4B1D49E6720DEE86463E915F6E90016AC  (= ggml-vocab-gpt-2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-gpt-neox.gguf  sha256=AE593A7F9B8BB174ED4F5019E41530463E4DAC7AA06E42DEE8AA650D2BDAC53D  (= ggml-vocab-gpt-neox.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-llama-bpe.gguf  sha256=97272E430D53BC7688F52D5E0AD8EA8F163EDE9F1BBD1694FEAA504797D5D96E  (= ggml-vocab-llama-bpe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-llama-spm.gguf  sha256=16C3724582D59AA8BF84711894E833F916EE46A31D80E21312759C48BF8D0E69  (= ggml-vocab-llama-spm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-mpt.gguf  sha256=59DC382612866D1FC6C11EA531318D327598F3412D9C8F8600607CDF3030898F  (= ggml-vocab-mpt.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-nomic-bert-moe.gguf  sha256=90A6746926454784A98389AD36A36D89BC9CFC81DB9CB0F33C941BCC959FE5F9  (= ggml-vocab-nomic-bert-moe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-phi-3.gguf  sha256=967D7190D11C4842EAB697079D98D56C2116E10EB617BE355A2733BFC132E326  (= ggml-vocab-phi-3.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-qwen2.gguf  sha256=44C2F46B715F585C6AB513970E8A006BFA5BADD6108560054921CF598D154D8C  (= ggml-vocab-qwen2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-qwen35.gguf  sha256=63ED952FF338996CF0BDF24A7B10015124273F75C6DC9BB427356AA3F67EC62C  (= ggml-vocab-qwen35.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-refact.gguf  sha256=AC3CEDA902FED91CCF74312B305D9B86C37E4F8E35FA9CC6EF3CE34FCA7D4678  (= ggml-vocab-refact.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\jusko-llama-volta-qwen3flash\models\ggml-vocab-starcoder.gguf  sha256=FEDB892B4E1BD3C1F2FCDAE356440B14FB458F4264D586E5C987ED93DF4E174D  (= ggml-vocab-starcoder.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-aquila.gguf  sha256=7C53C3C516AC67C7CA12977B9690FDEA3D2EF13BBAED6378F98191A13EF5CA00  (= ggml-vocab-aquila.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-baichuan.gguf  sha256=4F5B955697F3BD3108070B1D5936C7EB9FC542B81C6932E59ABDDEC75BCA1963  (= ggml-vocab-baichuan.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-bert-bge.gguf  sha256=FBCBE22278FB302694D5F4A41BFE48C5F90E8E3554EAB1C0435387DFF654A854  (= ggml-vocab-bert-bge.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-command-r.gguf  sha256=A2F8CFEA952EF7C391A6D92A1C309D0BD32E36384D9B9230569A7425732F27D9  (= ggml-vocab-command-r.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-deepseek-coder.gguf  sha256=91CB1379F2E33AF1C4866B194622B7A0E12E8F0C9DBA7BA2F10D55978730BEC1  (= ggml-vocab-deepseek-coder.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-deepseek-llm.gguf  sha256=867F77537B54565F0D81D508C04EDC41AA1D4FFC1A92745F225B4C1B02755F76  (= ggml-vocab-deepseek-llm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-falcon.gguf  sha256=9F0BF8B0733680398B72E652E90F260F43782F326E75545FC0E49611A5BA35AD  (= ggml-vocab-falcon.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-gemma-4.gguf  sha256=58B1BA0B57F3B4D7C468BA4FFD91AD85190346A3D7AD7E71D1CABAAE8A14BB65  (= ggml-vocab-gemma-4.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-gpt-2.gguf  sha256=CEDC56CA6E2E89F63E781696D1FD76B4B1D49E6720DEE86463E915F6E90016AC  (= ggml-vocab-gpt-2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-gpt-neox.gguf  sha256=AE593A7F9B8BB174ED4F5019E41530463E4DAC7AA06E42DEE8AA650D2BDAC53D  (= ggml-vocab-gpt-neox.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-llama-bpe.gguf  sha256=97272E430D53BC7688F52D5E0AD8EA8F163EDE9F1BBD1694FEAA504797D5D96E  (= ggml-vocab-llama-bpe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-llama-spm.gguf  sha256=16C3724582D59AA8BF84711894E833F916EE46A31D80E21312759C48BF8D0E69  (= ggml-vocab-llama-spm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-mpt.gguf  sha256=59DC382612866D1FC6C11EA531318D327598F3412D9C8F8600607CDF3030898F  (= ggml-vocab-mpt.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-nomic-bert-moe.gguf  sha256=90A6746926454784A98389AD36A36D89BC9CFC81DB9CB0F33C941BCC959FE5F9  (= ggml-vocab-nomic-bert-moe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-phi-3.gguf  sha256=967D7190D11C4842EAB697079D98D56C2116E10EB617BE355A2733BFC132E326  (= ggml-vocab-phi-3.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-qwen2.gguf  sha256=44C2F46B715F585C6AB513970E8A006BFA5BADD6108560054921CF598D154D8C  (= ggml-vocab-qwen2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-qwen35.gguf  sha256=63ED952FF338996CF0BDF24A7B10015124273F75C6DC9BB427356AA3F67EC62C  (= ggml-vocab-qwen35.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-refact.gguf  sha256=AC3CEDA902FED91CCF74312B305D9B86C37E4F8E35FA9CC6EF3CE34FCA7D4678  (= ggml-vocab-refact.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\sm70-attn\models\ggml-vocab-starcoder.gguf  sha256=FEDB892B4E1BD3C1F2FCDAE356440B14FB458F4264D586E5C987ED93DF4E174D  (= ggml-vocab-starcoder.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-aquila.gguf  sha256=7C53C3C516AC67C7CA12977B9690FDEA3D2EF13BBAED6378F98191A13EF5CA00  (= ggml-vocab-aquila.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-baichuan.gguf  sha256=4F5B955697F3BD3108070B1D5936C7EB9FC542B81C6932E59ABDDEC75BCA1963  (= ggml-vocab-baichuan.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-bert-bge.gguf  sha256=FBCBE22278FB302694D5F4A41BFE48C5F90E8E3554EAB1C0435387DFF654A854  (= ggml-vocab-bert-bge.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-command-r.gguf  sha256=A2F8CFEA952EF7C391A6D92A1C309D0BD32E36384D9B9230569A7425732F27D9  (= ggml-vocab-command-r.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-deepseek-coder.gguf  sha256=91CB1379F2E33AF1C4866B194622B7A0E12E8F0C9DBA7BA2F10D55978730BEC1  (= ggml-vocab-deepseek-coder.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-deepseek-llm.gguf  sha256=867F77537B54565F0D81D508C04EDC41AA1D4FFC1A92745F225B4C1B02755F76  (= ggml-vocab-deepseek-llm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-falcon.gguf  sha256=9F0BF8B0733680398B72E652E90F260F43782F326E75545FC0E49611A5BA35AD  (= ggml-vocab-falcon.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-gemma-4.gguf  sha256=58B1BA0B57F3B4D7C468BA4FFD91AD85190346A3D7AD7E71D1CABAAE8A14BB65  (= ggml-vocab-gemma-4.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-gpt-2.gguf  sha256=CEDC56CA6E2E89F63E781696D1FD76B4B1D49E6720DEE86463E915F6E90016AC  (= ggml-vocab-gpt-2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-gpt-neox.gguf  sha256=AE593A7F9B8BB174ED4F5019E41530463E4DAC7AA06E42DEE8AA650D2BDAC53D  (= ggml-vocab-gpt-neox.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-llama-bpe.gguf  sha256=97272E430D53BC7688F52D5E0AD8EA8F163EDE9F1BBD1694FEAA504797D5D96E  (= ggml-vocab-llama-bpe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-llama-spm.gguf  sha256=16C3724582D59AA8BF84711894E833F916EE46A31D80E21312759C48BF8D0E69  (= ggml-vocab-llama-spm.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-mpt.gguf  sha256=59DC382612866D1FC6C11EA531318D327598F3412D9C8F8600607CDF3030898F  (= ggml-vocab-mpt.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-nomic-bert-moe.gguf  sha256=90A6746926454784A98389AD36A36D89BC9CFC81DB9CB0F33C941BCC959FE5F9  (= ggml-vocab-nomic-bert-moe.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-phi-3.gguf  sha256=967D7190D11C4842EAB697079D98D56C2116E10EB617BE355A2733BFC132E326  (= ggml-vocab-phi-3.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-qwen2.gguf  sha256=44C2F46B715F585C6AB513970E8A006BFA5BADD6108560054921CF598D154D8C  (= ggml-vocab-qwen2.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-refact.gguf  sha256=AC3CEDA902FED91CCF74312B305D9B86C37E4F8E35FA9CC6EF3CE34FCA7D4678  (= ggml-vocab-refact.gguf)  同哈希已收录
- DUP  F:\vllm+llama.cpp\v100-refs\xllama.cpp\models\ggml-vocab-starcoder.gguf  sha256=FEDB892B4E1BD3C1F2FCDAE356440B14FB458F4264D586E5C987ED93DF4E174D  (= ggml-vocab-starcoder.gguf)  同哈希已收录

## 六、源删除确认 (2026-09-23T11:36:19.7142987+08:00)
- 已删根目录: QWEN.md, ARCHIVE-COMMIT-MSG.txt, local-lf-ggml-cuda.cu, .sshprobe.txt(.err), 3 个 redirect 垃圾文件, .qwen/, .tmpcmp/
- 已删源 GGUF 75 处（去重后 19 个已入 models/vocab）
- **保留** 根目录 AGENTS.md（在役）
- 根目录现为: docs/ models/ olddoc/ + 代码与研究目录
