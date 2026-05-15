# SGLang Issue #24335 调研笔记

**Issue 标题**：`[Refactor] Pool / Allocator module restructure under mem_cache/`
**作者**：@hnyls2002 ｜**创建时间**：2026-05-04 ｜**状态**：open ｜**标签**：`roadmap`、`Refactor`
**链接**：https://github.com/sgl-project/sglang/issues/24335
**并行 roadmap**：#20415 `[Roadmap] Unified Hybrid Radix Cache Refactor`（同模块、不同分层）

---

## 一、讨论的问题

`python/sglang/srt/mem_cache/` 目录下的 **pool 层**（KV cache 物理存储）与 **allocator 层**（token-to-KV 索引分配）存在结构性问题，**与 #20415 中的 radix tree 统一工作相互独立**：

1. **类型边界不清**：一个文件里塞多个互不相关的类，跨 family 的类型耦合在一起，新手难以从目录结构看出 "MHA / MLA / Mamba / SWA / DSv4 / HiSparse" 各自的归属。
2. **feature scope 渗漏**：DSv4 与 HiSparse 两个特性的类互相塞到对方的文件里，违反 "一个文件 = 一个特性域" 的边界。
3. **子模块命名规则不一致**：现存的两个子目录采用两种风格，没有统一约定。

Issue 强调："**纯 vibe coding PR 和未充分讨论的 naïve 改动不欢迎**" —— 所有改动需配合 owner 讨论。

---

## 二、背景

### 2.1 当前 `mem_cache/` 目录现状（实例）

```
mem_cache/
├── allocator.py                       # 单文件，3 个类
├── memory_pool.py                     # 单文件，11 个类，~2000 行
├── memory_pool_host.py                # 单文件，host 侧 8 个类
├── deepseek_v4_memory_pool.py         # DSv4 文件
├── deepseek_v4_compress_state.py
├── hisparse_memory_pool.py            # HiSparse 文件
├── swa_memory_pool.py
├── base_swa_memory_pool.py
├── mamba_radix_cache.py
├── hi_mamba_radix_cache.py
├── radix_cache.py / swa_radix_cache.py / unified_radix_cache.py / ...
├── hybrid_cache/                      # 子模块（保留前缀 hybrid_）
│   ├── hybrid_cache_controller.py
│   └── hybrid_pool_assembler.py
├── unified_cache_components/          # 子模块（去掉前缀，用 _components 后缀）
│   ├── full_component.py
│   ├── mamba_component.py
│   ├── swa_component.py
│   └── tree_component.py
└── ...
```

### 2.2 三类结构性问题的具体证据

#### ① `allocator.py` 单文件包多个 family —— 应升级为模块

```python
# python/sglang/srt/mem_cache/allocator.py
class BaseTokenToKVPoolAllocator(abc.ABC):     # L35
class TokenToKVPoolAllocator(...):              # L121  连续分配
class PagedTokenToKVPoolAllocator(...):         # L362  paged 分配
```
另有 `hisparse_memory_pool.py` 中独立的 `HiSparseTokenToKVPoolAllocator`、`DeepSeekV4HiSparseTokenToKVPoolAllocator` 属同一抽象族，却散落在别处。

#### ② `memory_pool.py` 是 catch-all —— 应按 family 拆包

```python
# python/sglang/srt/mem_cache/memory_pool.py（~2000 行）
class ReqToTokenPool:                # L127  req → token 索引
class MambaPool:                     # L194  Mamba state pool
class HybridReqToTokenPool(...):     # L486
class KVCache(abc.ABC):              # L692  KV 抽象基类
class MHATokenToKVPool(KVCache):     # L788  MHA
class NoOpMHATokenToKVPool(...):     # L1135
class MHATokenToKVPoolFP4(...):      # L1245
class HybridLinearKVPool(...):       # L1388 hybrid 线性
class MLATokenToKVPool(KVCache):     # L1617 MLA
class MLATokenToKVPoolFP4(...):      # L1851
class NSATokenToKVPool(...):         # L1980 NSA
```

`memory_pool_host.py` 同样塞了 MHA / MLA / Mamba / NSA / PoolGroup 等 8 个类，需要同步镜像拆分。

#### ③ DSv4 ↔ HiSparse 跨域污染（最关键的一项）

```python
# deepseek_v4_memory_pool.py  —— DSv4 文件却出现 HiSparse 类
class DeepSeekV4SingleKVPool(KVCache):           # L44
class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):  # L159  ❌ HiSparse 漏到 DSv4 文件
class DeepSeekV4IndexerPool(KVCache):            # L242
class DeepSeekV4TokenToKVPool(BaseSWAKVPool):    # L351
```

```python
# hisparse_memory_pool.py  —— HiSparse 文件却出现 DSv4 类
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (...)  # cross-import
class HiSparseNSATokenToKVPool(NSATokenToKVPool):              # L39
class HiSparseTokenToKVPoolAllocator(...):                     # L138
class DeepSeekV4SingleKVPoolHost:                              # L388  ❌ DSv4 漏到 HiSparse 文件
class DeepSeekV4HiSparseTokenToKVPoolAllocator(...):           # L503  ❌ 混合命名混合归属
```

两个文件互相 import 对方的核心符号，导致 feature 边界完全失效。

#### ④ 子模块命名风格冲突

| 现存目录 | 命名风格 |
|---|---|
| `hybrid_cache/` | 保留前缀（`hybrid_cache_controller`, `hybrid_pool_assembler`） |
| `unified_cache_components/` | 去掉前缀、用 `_component` 后缀（`full_component`, `mamba_component`, ...） |

需要全仓库统一一种。

---

## 三、Work Items（来自 issue 原文）

- [ ] **把 `allocator` 升级为独立 module** —— one file per family
- [ ] **把 `pool` 升级为独立 module** —— 用 per-family 子包替换 catch-all 的 `memory_pool.py`
- [ ] **host 侧做同样的拆分**（当前在 `memory_pool_host.py`）
- [ ] **解开 DSv4 / HiSparse 的交叉污染** —— 各自的类回到正确的 feature 文件，禁止跨 feature 文件 import
- [ ] **统一一种子模块命名规则** —— `hybrid_cache/` 保留前缀 vs `unified_cache_components/` 去掉前缀，全仓择一
- [ ] 上述落地后，**补一份 `(type × scope) → location` 的索引文档**

### 优先级

> Allocator module 化 + DSv4/HiSparse 解耦优先（blast radius 最小）。
> Pool / host-pool 模块化更大，每项各自一个 PR。

---

## 四、检查 / 验证方案

### 4.1 准入门槛（开工前对齐）

1. 在 issue 下评论确认拆分粒度、目录命名（必须二选一统一）、是否引入 `__init__.py` 重导出做兼容层；务必在 owner（@hnyls2002 / @ispobock）确认后再动手。
2. 每项 work item 单独开 PR，避免一次性大 diff。

### 4.2 静态检查项（每个 PR 都要过）

| 检查点 | 命令 / 方式 | 期望 |
|---|---|---|
| 跨 feature 文件不 import | `grep -R "from sglang.srt.mem_cache.deepseek_v4" python/sglang/srt/mem_cache/hisparse*` 与反向 | 拆分后应**为空** |
| 类与文件 1:1 对应 | `grep -nE "^class " python/sglang/srt/mem_cache/<file>` | 每个 family 文件只剩本 family 的类 |
| 子模块命名一致 | `ls python/sglang/srt/mem_cache/` | 全部前缀风格或全部 `_components` 后缀风格 |
| 外部引用全部跟随更新 | `grep -R "from sglang.srt.mem_cache.memory_pool import" python/ test/` | 命中处全部 import 自新位置；旧路径若保留则只在 `__init__.py` 做 alias 重导出 |
| host 镜像同步 | 与 device 侧文件树 diff | 结构同形 |

### 4.3 行为等价性验证

- **单元测试**：`test/srt/test_radix_cache.py`、`test/srt/test_hicache_storage.py`、`test/srt/test_swa_*`、`test/srt/test_mamba_*`、`test/srt/test_deepseek_v4_*`、`test/srt/test_hisparse_*`（按拆到的 feature 跑相应子集）。
- **CI smoke**：触发 `pr-test.yml`、`pr-test-h100.yml` 的 mem_cache 相关用例。
- **运行时一致性**：refactor PR 必须是**纯结构调整**，不允许引入逻辑 diff —— 用 `git diff --stat -M -B` 看是否主要是 rename / move，再 `git diff -M` 复核搬运后的代码字节级未变。
- **下游兼容**：搜索 `python/sglang/srt/managers/`、`python/sglang/srt/disaggregation/`、`python/sglang/srt/speculative/`、`docs/`、外部已发的 example，确认所有引用 mem_cache 类的位置都被同步更新。

### 4.4 文档与索引

- 末轮 PR 提交一份 `python/sglang/srt/mem_cache/README.md` 类型索引（type × scope → file），示例形态：

| type \ scope | full | swa | mamba | mla | nsa | dsv4 | hisparse |
|---|---|---|---|---|---|---|---|
| allocator | `allocator/token.py` | `allocator/paged.py` | ... | ... | ... | ... | `allocator/hisparse.py` |
| pool (device) | `pool/mha.py` | `pool/swa.py` | `pool/mamba.py` | `pool/mla.py` | `pool/nsa.py` | `pool/dsv4.py` | `pool/hisparse.py` |
| pool (host) | `pool_host/mha.py` | ... | ... | ... | ... | ... | ... |

（最终目录形态由 owner 拍板）

---

## 五、与 #20415 的边界（避免混淆）

- **#20415**：radix tree / 缓存 **policy 层**（RadixCache、MambaRadixCache、SWARadixCache、UnifiedRadixTree …）的统一与去重。
- **#24335（本 issue）**：tree 之下的 **物理 pool 层 + allocator 层** 的目录与文件级重组，不涉及树结构语义。
- 两条线并行推进，PR 拆分时**避免同 PR 同时改两层**，否则 review 难度爆炸。
