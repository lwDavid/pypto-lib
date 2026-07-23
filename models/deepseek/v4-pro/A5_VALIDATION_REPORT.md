# DeepSeek V4-Pro A5 全量算子验证报告(int8 基线)

- **日期**: 2026-07-23
- **范围**: `models/deepseek/v4-pro/` 全部 35 个可运行算子(`__main__`,排除 `*draft*`)
- **平台**: Ascend 950 (A5),设备卡 `2,3,4,5`(`task-submit --device auto` 自动分配,避开坏卡 0/1/7)
- **结论**: **32 通过 / 3 失败(通过率 91.4%)**。3 个失败正是预期的「prefill indexer/layer/fwd」。

## 生效的组合(pto-1 配套,已在 pto-2 复刻)

| 组件 | 版本 | 说明 |
|------|------|------|
| pypto | `86fec741` + **a5 guard 补丁** | 单步 cast(避免主线 cast bug);本地补丁在 a5 上跳过 `set_allow_early_resolve` |
| simpler (runtime) | `41fc8ba8` | 二进制 + nanobind 扩展**一致**重建(见「关键修复」) |
| pypto-lib | `3372ad9` | pto-1 的 v4-pro 版本(`fix/hc_pre-a5-guard` 分支,从 pto-1-local fetch) |
| pto-isa | `83d01313` | simpler `pto_isa.pin` |
| ptoas | `0.48` | `~/pto-2/ptoas-bin` |
| CANN | `cann-9.1.T500` | `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.T500` |

> 该组合等价于 pto-1 历史上跑通 32/35 的配套状态。

## 关键修复(过程中踩的坑)

1. **cast bug(主线 pypto)**:主线 pypto 把量化降级成 `float→half→int8` 两步,新增的 `half←float` cast 命中 pto-isa `tcvt_common.hpp` 的 `<R>` 模板分派 bug(A5 CANN 不支持)→ 全部 cast 类算子编译失败。**用 pypto `86fec741`(单步 cast)绕开。**
2. **L3 接口不匹配**:pypto `86fec741` 的旧 L3 API 与新 simpler `d4071fe1` 不匹配(`Worker: add_worker after init`)→ 多卡算子全挂。**simpler 对齐到 `41fc8ba8`。**
3. **simpler 扩展/二进制不一致(我的设置失误)**:切 simpler 到 `41fc8ba8` 时只跑了 `build_runtimes.py`(二进制),忘了 `pip install -e .`(nanobind 扩展),导致扩展(旧 d4071fe1,会 dlsym `simpler_provision_dma_workspace`)和二进制(41fc8ba8,无该符号)不一致 → `dlsym undefined symbol`。**补跑 `pip install --no-build-isolation --no-deps -e .` 重建扩展,一致后消失。**
4. **`set_allow_early_resolve` 在 a5 缺失**:simpler `41fc8ba8` 的 a5 `L0TaskArgs` 没有该方法 → orchestration 编译失败。**pypto 打 a5 guard(仅 a5 跳过该调用)。**

> 教训:切换 simpler 版本时,**`build_runtimes.py`(二进制)和 `pip install -e .`(nanobind 扩展)都要重跑**,二者必须一致。

## 通过的算子(32 个)

| 算子 | perf(μs) | | 算子 | perf(μs) |
|------|----------|---|------|----------|
| decode_attention_csa | 5210.4 | | prefill_attention_csa | - |
| decode_attention_hca | 483.6 | | prefill_attention_hca | 2159.3 |
| decode_attention_swa | 375.6 | | prefill_attention_swa | 2117.2 |
| decode_compressor_ratio128 | 160.6 | | prefill_compressor_ratio128 | 231.0 |
| decode_compressor_ratio4 | 128.6 | | prefill_compressor_ratio4 | 205.2 |
| decode_fwd (2卡) | 53731.0 | | prefill_mtp (2卡) | 32315.9 |
| decode_indexer | 268.8 | | prefill_indexer_compressor | 359.4 |
| decode_indexer_compressor | 1543.4 | | prefill_sparse_attn | - |
| decode_layer (2卡) | 12642.9 | | qkv_proj_rope | 1048.7 |
| decode_mtp (2卡) | 10942.3 | | rmsnorm | 29.7 |
| decode_sparse_attn | 232.3 | | expert_routed | 453.2 |
| decode_sparse_attn_hca * | - | | expert_shared | 64.0 |
| decode_sparse_attn_swa | 199.6 | | gate | 465.2 |
| lm_head (2卡) | 95620.3 | | hc_head | 36.6 |
| moe (2卡) | 4493.6 | | hc_post | 15.1 |
| mtp_projection | 3077.1 | | hc_pre | 99.1 |

\* `decode_sparse_attn_hca`、`prefill_sparse_attn` 在并发扫描(并发4 + 共享机器其他用户占卡)下偶发 AICPU sync 超时(507000/507018);**串行独占设备重跑均 PASS**,属负载偶发,非真实失败。

## 失败的算子(3 个,均为预期)

| 算子 | 卡数 | 失败原因 |
|------|------|----------|
| `prefill_indexer.py` | 1 | 数值校验:`'score'` 不过(ratio_allclose 超 threshold),其余输出 PASS |
| `prefill_fwd.py` | 2 | AICPU sync 超时(507000),整段 prefill forward 过重 |
| `prefill_layer.py` | 2 | AICPU sync 超时 |

> 这 3 个即 pto-1 历史上的「最好情况只失败这 3 个」。

## 复现

```bash
# 组合:pypto 86fec741(+a5 guard)、simpler 41fc8ba8(二进制+扩展一致)、pypto-lib 3372ad9、pto-isa 83d01313
# 扫描脚本:/tmp/v4pro_a5_val/run_sweep.sh(并发 4,task-submit --device auto,多卡 --device-num N)
# 每算子日志:/tmp/v4pro_a5_val/logs/<op>.log  结果:/tmp/v4pro_a5_val/results/<op>.tsv
```

## 下一步

int8 基线已就位(32/35)。下一步「向 Pro 量化策略对齐」:按文档在 A5 上把 v4-pro 量化从 int8 改为 **FP8 系(MXFP8/MXFP4/HiF8,KV Cache FP8 C8)**。预期 cast 目标转为 `float8_e4m3_t`/`hifloat8_t` 等(A5 原生,绕开 int8/half cast bug),且可在最新栈(pypto main + simpler d4071fe1 + pypto-lib dcdd061)上进行。
