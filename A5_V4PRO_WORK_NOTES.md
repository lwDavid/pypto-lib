# A5 V4-Pro 工作笔记(配套版本 + 成果 + 关键发现)

> 用途:记录在 Ascend 950 (A5) 上跑通 DeepSeek V4-Pro 的**可用配套版本组合**、**已完成的修复**、以及排查中得到的**关键发现**,便于环境迁移/复现。日期:2026-07-23。

## 1. 可用配套版本组合(int8,验证 32/35 通过)

在 A5(`cann-9.1.T500`)上跑通 v4-pro 32/35 的组合(等价 pto-1 历史 benchmark 配套):

| 组件 | commit | 说明 |
|------|--------|------|
| pypto | `86fec741` + **a5 guard 补丁**(见 §3) | 单步 cast,绕开主线 cast bug(§4);已推 `lwDavid/pypto` 分支 |
| simpler (runtime) | `41fc8ba8` | a5 分支那版;L3 API 与 pypto 86fec741 配套 |
| pypto-lib | `3372ad9` + **indexer 修复**(见 §2) | pto-1 的 v4-pro;已推 `lwDavid/pypto-lib` 分支 |
| pto-isa | `83d01313` | simpler `pto_isa.pin` |
| ptoas | `0.48` | `~/pto-2/ptoas-bin` |
| CANN | `cann-9.1.T500` | `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.T500` |

**结果**:32 pass / 3 fail。3 个失败 = 预期的 `prefill_indexer`(数值,§2 已修)、`prefill_fwd`、`prefill_layer`(大 prefill forward,AICPU sync 超时)。

## 2. prefill_indexer 修复(本次核心成果)

**根因(不是 isa bug)**:`prefill_indexer.py` score 段的 `qh_scale_s` reshape 非法:
```python
# 原 bug:IDX_N_HEADS(=64)个元素 reshape 成 [1, IDX_HEAD_DIM=128] —— 非法
qh_scale_s = pl.reshape(qr_hadamard_scale_dq[q_s0:q_s0+IDX_N_HEADS, :], [1, IDX_HEAD_DIM])
# 修复(对齐 decode_indexer):
qh_scale_s = pl.reshape(qr_hadamard_scale_dq[q_s0:q_s0+IDX_N_HEADS, :], [1, IDX_N_HEADS])
```
- 原 fused kernel 里被 codegen 掩盖(没报错但 qh_scale 错)→ score 数值错。
- decode_indexer 用正确的 `[1, IDX_N_HEADS]` 所以通过 → **同一 int8 TMATMUL 没问题,不是 isa bug**。

**附带改动**:把 score 从单 fused kernel 拆成两段(对齐 decode 的 `score_mat`/`score_reduce`):
- `score_acc_gm = pl.create_tensor([T*INDEXER_SCORE_CAP, IDX_N_HEADS], INT32)`(GM 中转)
- `prefill_idx_score_mat`(cube: int8 matmul → INT32 写 GM)
- `prefill_idx_score_reduce`(vector: cast/dequant/relu/加权 row_sum → score_wide)

> 拆分后 codegen 不再掩盖 reshape,直接报错暴露了真 bug。**结构已验证可编译**(一次 run 跑到编译、只在 reshape 报错;改完 reshape 待设备恢复验证 score 是否 PASS)。

文件:`models/deepseek/v4-pro/prefill_indexer.py`。详见 `models/deepseek/v4-pro/A5_VALIDATION_REPORT.md`。

## 3. pypto a5 guard 补丁

`src/codegen/orchestration/orchestration_codegen.cpp` 的 `EmitEarlyResolveHint`:在 a5(Ascend950)上跳过 `set_allow_early_resolve`(simpler 41fc8ba8 的 a5 `L0TaskArgs` 没有此方法)。
```cpp
if (call->GetAttr<bool>("allow_early_resolve", false) &&
    pypto::backend::GetBackendType() != pypto::backend::BackendType::Ascend950) {
  EmitIndentedLine(task_var + ".set_allow_early_resolve(true);");
}
```
已推 `lwDavid/pypto`(基于 `86fec741`)。

## 4. cast bug(为何不能用最新 pypto 主线)

主线 pypto 把量化降级成两步 `float→half→int8`,新增的 `half←float` cast 命中 pto-isa `tcvt_common.hpp` 的 `<R>` 模板分派 bug(A5 CANN 不支持)→ 所有 cast 类算子编译失败。**无 isa 版本可修**(83d01313 用 `<R>`、重构前 269666ba 缺 half 重载)。故回退到单步 cast 的 pypto `86fec741`。

## 5. 版本联动坑(切换 simpler 的教训)

切 simpler 版本时,**`build_runtimes.py`(runtime 二进制)和 `pip install -e .`(nanobind 扩展)必须一起重跑**,否则扩展(旧)和二进制(新)不一致 → `dlsym simpler_provision_dma_workspace undefined symbol`。

主线 simpler/pypto 的 cast-regression + L3-API + SDMA 符号是**绑死**演进的,无法和新老版本混搭出全通组合;只能用 §1 的配套全套。

## 6. 环境运行要点

- 设备借卡:`task-submit --device auto`(本机 0/1/7 号卡损坏,白名单 2/3/4/5)。
- 用户不在 HwHiAiUser 组,必须走 task-submit(root 队列)借卡,不能直接 `-d`。
- ptoas 需 `LD_LIBRARY_PATH=$PTOAS_ROOT/lib`(否则 `libMLIRMlirOptMain.so.19.1` 缺失)。
- 运行环境变量:`PYPTO_LOG_LEVEL=error PYPTO_WARNING_LEVEL=none PYPTO_RUNTIME_LOG=error PYPTO_BENCH=1 PTO2_RING_DEP_POOL=16384 PTO2_RING_TASK_WINDOW=16384 PTO2_RING_HEAP=1073741824`。

## 7. 遗留 / 风险

- **设备 107001**:排查期间 A5 设备出现稳定 `rtSetDevice failed: 107001`(所有卡、连 rmsnorm 都 init 失败),需管理员 reset 驱动。indexer 修复的 score PASS 验证因此 pending。
- `prefill_fwd`/`prefill_layer` 的 AICPU sync 超时(大 prefill forward)未修。
- pypto-lib 的修改基于 `3372ad9`(pto-1 分支),合入主线(dcdd061)时需确认 prefill_indexer.py 是否一致(预期一致)。
