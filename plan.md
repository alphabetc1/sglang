# mem_cache Refactor 推进计划（Issue #24335）

> 背景与问题诊断详见 `issue.md`。本文聚焦：**优先级、整体路线图、PR 切分策略**。

---

## 一、优先级建议（TL;DR）

按 **"blast radius 从小到大、依赖前置项在前"** 排序：

| 序号 | 任务 | blast radius | 风险 | 建议 PR 大小 |
|---|---|---|---|---|
| **P0** | DSv4 ↔ HiSparse 跨域污染解耦 | 小（局部 2 文件 + 少量下游 import） | 低 | 1 个 PR |
| **P0** | `allocator.py` → `allocator/` 模块化 | 小（单文件，3 类 + HiSparse / DSv4 几个迁入） | 低 | 1 个 PR |
| **P1** | 子模块命名风格统一（`hybrid_cache/` vs `unified_cache_components/`） | 中（纯 rename，但牵动 import） | 低 | 1 个 PR |
| **P2** | `memory_pool.py` → `pool/` per-family 拆包 | 大（2000 行 / 11 类 / 大量下游 import） | 中 | 1 个 PR（必要时按 family 二次切，但避免半态） |
| **P2** | `memory_pool_host.py` → `pool_host/` 镜像拆包 | 中（与 P2 同构，依赖 P2 命名） | 中 | 1 个 PR |
| **P3** | `(type × scope) → location` 索引文档 | 极小 | 无 | 1 个 PR |

**理由（与 issue 原文一致）**：作者 hnyls2002 已经明确 _"Allocator module + DSv4/HiSparse untangling first (smallest blast radius)"_，所以 P0 选择服从 owner 设定。Pool / host-pool 拆分必须各自独立 PR。

---

## 二、整体路线图

### Phase 0：对齐（不写代码，但是必须做）

- 在 issue #24335 下评论提议本计划，**等 owner（@hnyls2002 / @ispobock）确认这几点**：
  1. 子模块命名最终风格：**前缀派**（`hybrid_cache/`）还是**后缀派**（`unified_cache_components/`）？
  2. allocator / pool 包内部模块切分维度：按 **family**（mha/mla/mamba/nsa/swa/dsv4/hisparse）切，还是按 **抽象层**（base/contiguous/paged）切？我倾向 family，与 issue "one file per family" 字面一致。
  3. 旧 import 路径是否保留 alias 过渡（建议保留 1～2 个 release，给外部插件迁移窗口）。
- **没拿到回复前不动手**，避免 vibe coding。

### Phase 1：P0 —— 小、快、稳的两个 PR（可并行）

#### PR-A：DSv4 / HiSparse 跨文件污染清理（**最先合**）

**问题（来自 issue.md）**：
- `deepseek_v4_memory_pool.py:159` 出现 `HiSparseC4DevicePool`（继承 `DeepSeekV4SingleKVPool`，但语义属 HiSparse）
- `hisparse_memory_pool.py:388` 出现 `DeepSeekV4SingleKVPoolHost`
- `hisparse_memory_pool.py:503` 出现 `DeepSeekV4HiSparseTokenToKVPoolAllocator`
- 两文件互相 `import` 对方核心类

**动作**：纯搬运 + import 路径修正
```
deepseek_v4_memory_pool.py:
  - HiSparseC4DevicePool                        → hisparse_memory_pool.py
hisparse_memory_pool.py:
  - DeepSeekV4SingleKVPoolHost                  → deepseek_v4_memory_pool.py（或拆出 dsv4 host 文件，依 P2 决策）
  - DeepSeekV4HiSparseTokenToKVPoolAllocator    → 留在 hisparse_memory_pool.py（HiSparse 适配器逻辑）
                                                  但要去掉对 dsv4 内部细节的反向依赖
```

**验收**：
- `grep "from .*deepseek_v4_memory_pool" python/sglang/srt/mem_cache/hisparse_memory_pool.py` 仅剩对纯数据类 / 类型签名的 import，**不再 import 实现类**
- 反向 grep 应该为空
- 单测：DSv4 与 HiSparse 各自的 e2e（按 `test/srt/test_*deepseek_v4*` / `test/srt/test_*hisparse*` 全跑）

---

#### PR-B：`allocator.py` → `allocator/` 模块化

**当前结构**：
```python
# allocator.py (单文件)
class BaseTokenToKVPoolAllocator(abc.ABC):  # L35
class TokenToKVPoolAllocator(...):           # L121
class PagedTokenToKVPoolAllocator(...):      # L362
```
另有散落在 `hisparse_memory_pool.py` 的 `HiSparseTokenToKVPoolAllocator`、`DeepSeekV4HiSparseTokenToKVPoolAllocator`（后者由 PR-A 落定归属后再纳入）。

**目标结构**：
```
mem_cache/allocator/
├── __init__.py        # re-export 全部 public 类，保持 from sglang.srt.mem_cache.allocator import X 仍可用
├── base.py            # BaseTokenToKVPoolAllocator
├── contiguous.py      # TokenToKVPoolAllocator（连续）
├── paged.py           # PagedTokenToKVPoolAllocator（分页）
└── hisparse.py        # HiSparseTokenToKVPoolAllocator + DeepSeekV4HiSparse... （PR-A 合并后再合）
```

**关键约束**：
- `__init__.py` 必须把所有原 public 符号 re-export，**短期内不动外部 import 路径**，让 PR diff 局限在 mem_cache/ 内部。
- 下一轮单独 PR 再去外部把 import 路径切到 `from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator`（如果决定切的话）。

**验收**：
- `python -c "from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator, TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator"` 通过
- `git diff -M -B --stat` 主体应是 rename / move，没有逻辑改动
- CI 全量过

---

### Phase 2：P1 —— 命名风格统一

#### PR-C：rename `unified_cache_components/` 或 `hybrid_cache/` 二选一

依据 Phase 0 owner 拍板的风格，做单纯目录 rename + 文件 rename：

**方案 1（推荐：去前缀派，符合 "目录已是 scope，文件不必再重复"）**：
```
unified_cache_components/{full,swa,mamba,tree}_component.py  保持不变
hybrid_cache/hybrid_cache_controller.py    → hybrid_cache/controller.py
hybrid_cache/hybrid_pool_assembler.py      → hybrid_cache/pool_assembler.py
```

**方案 2（保留前缀派）**：
```
unified_cache_components/  → 重新审视；component 后缀对 tree_component 不太自然
```

**验收**：
- 仅 rename，无逻辑改动；`git log --follow` 能追溯历史
- 全仓 grep 旧路径无残留：`grep -R "hybrid_cache_controller\|unified_cache_components" python/ test/`

---

### Phase 3：P2 —— Pool 与 Host-Pool 拆包（最大动土）

#### PR-D：`memory_pool.py` → `pool/` per-family

**当前**（`memory_pool.py` ~2000 行，11 类）：
| 类 | 行号 | 目标位置 |
|---|---|---|
| `ReqToTokenPool`, `HybridReqToTokenPool` | L127, L486 | `pool/req_to_token.py` |
| `MambaPool` | L194 | `pool/mamba.py` |
| `KVCache` (abc) | L692 | `pool/base.py` |
| `MHATokenToKVPool`, `NoOpMHATokenToKVPool`, `MHATokenToKVPoolFP4` | L788, L1135, L1245 | `pool/mha.py` |
| `HybridLinearKVPool` | L1388 | `pool/hybrid_linear.py` |
| `MLATokenToKVPool`, `MLATokenToKVPoolFP4` | L1617, L1851 | `pool/mla.py` |
| `NSATokenToKVPool` | L1980 | `pool/nsa.py` |

**额外**（依赖 PR-A 解耦完）：
- `deepseek_v4_memory_pool.py` 整体迁入 `pool/dsv4.py`
- `swa_memory_pool.py` + `base_swa_memory_pool.py` 合并入 `pool/swa.py`（或 `pool/swa/{base,impl}.py`）
- `hisparse_memory_pool.py` 的 pool 部分迁入 `pool/hisparse.py`，allocator 部分已在 PR-B 处理

**`pool/__init__.py` re-export 保留**，给外部插件 1～2 release 的过渡。

**PR 大小控制**：单 PR 完成（避免半态目录结构）。如果 diff 过大，可以**先 mechanical move 再做 import 整理**，但仍是同一个 PR；不要把 mechanical move 拆到多个 PR，否则中间态难 review。

**验收**：
- `git diff -M -B --stat` 主体仍是 rename
- 关键 e2e：`MHA/MLA/Mamba/NSA/SWA/DSv4/HiSparse` 各自的 server smoke（CI 上对应 workflow 全过）
- benchmark 抽样：随便一个常用模型起服务跑 5 个请求，对比 PR 合并前后 token 输出 bit-for-bit 一致（保险）

---

#### PR-E：`memory_pool_host.py` → `pool_host/`（镜像 PR-D）

依赖 PR-D 已合，**目录树同构**：
```
pool_host/
├── __init__.py
├── base.py            # HostKVCache, HostTensorAllocator
├── mha.py             # MHATokenToKVPoolHost
├── mla.py             # MLATokenToKVPoolHost
├── mamba.py           # MambaPoolHost
├── nsa.py             # NSAIndexerPoolHost
├── group.py           # PoolEntry, HostPoolGroup
└── dsv4.py            # DeepSeekV4SingleKVPoolHost（PR-A 后归位至此）
```

**为什么单独 PR**：device / host 拆分如果合并到 PR-D，一次 diff 太大，review 注意力分散；CI runner 也会更慢。

---

### Phase 4：P3 —— 索引文档

#### PR-F：`mem_cache/README.md` 加 (type × scope) → location 表

| type \ scope | full(MHA) | mla | mamba | nsa | swa | hybrid_linear | dsv4 | hisparse |
|---|---|---|---|---|---|---|---|---|
| allocator | `allocator/contiguous.py` | — | — | — | `allocator/paged.py` | — | — | `allocator/hisparse.py` |
| pool (device) | `pool/mha.py` | `pool/mla.py` | `pool/mamba.py` | `pool/nsa.py` | `pool/swa.py` | `pool/hybrid_linear.py` | `pool/dsv4.py` | `pool/hisparse.py` |
| pool (host) | `pool_host/mha.py` | `pool_host/mla.py` | `pool_host/mamba.py` | `pool_host/nsa.py` | — | — | `pool_host/dsv4.py` | — |
| req-to-token | `pool/req_to_token.py` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ |

（表格随实际拆分微调；同 PR 顺手补 module-level docstring）

---

## 三、PR 切分总图

```
Phase 0  对齐（issue 评论）
   │
   ├─► PR-A: DSv4/HiSparse 解耦      ────┐
   ├─► PR-B: allocator/ 模块化       ────┤  P0，可并行；建议 PR-A 先合，PR-B 再合
   │                                     │  (PR-B 末尾才合 HiSparseAllocator 类)
   ▼                                     ▼
PR-C: 子模块命名风格统一  (P1，单独合)
   │
   ▼
PR-D: pool/ 拆包  (P2，最大)
   │
   ▼
PR-E: pool_host/ 拆包  (P2，依赖 D)
   │
   ▼
PR-F: README 索引文档  (P3，收尾)
```

**总计 6 个 PR**，按依赖串行 + 部分并行；预计总跨度 2～3 个 release 周期（因为每个 PR 都需要 owner review 与 CI 全过）。

---

## 四、共通约束（每个 PR 都要遵守）

1. **纯结构改动，零逻辑 diff**。用 `git diff -M -B` 检查搬运后是否字节级未变。
2. **PR 描述带上 issue link**（`Refs #24335`），并列出本 PR 处理的具体 work item bullet。
3. **保留 `__init__.py` re-export 至少一个 release**，外部插件作者有迁移窗口。
4. **不混入** radix tree / cache policy 层修改（那是 #20415 的轨道），review 才不会爆炸。
5. **不在 refactor PR 里顺手"优化"**（包括给函数加 type hint、改 logger、调 default 参数）—— 单独 PR。
6. CI 必须全绿；任何 flaky test 用既有规则单独开 issue，不在本 PR 里 mark skip。
