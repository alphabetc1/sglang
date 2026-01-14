# 动态切换 Disaggregation Mode 设计方案

## 一、背景与目标

### 1.1 背景

SGLang 当前支持 PD Disaggregation（Prefill-Decode 分离）架构，通过 `--disaggregation-mode` 参数在启动时指定节点角色：
- `null`: 普通模式，同时执行 Prefill 和 Decode
- `prefill`: 仅执行 Prefill，将 KV Cache 传输给 Decode 节点
- `decode`: 仅执行 Decode，接收来自 Prefill 节点的 KV Cache

当前设计中，模式在启动时确定且无法更改，这限制了集群的灵活性和资源利用率。

### 1.2 目标

实现动态切换 `--disaggregation-mode` 的能力：
1. 支持 NULL ↔ PREFILL ↔ DECODE 三种模式之间的动态切换
2. 支持非 disaggregation 节点（NULL）动态转换为 disaggregation 节点（PREFILL/DECODE）
3. 切换过程安全可靠，支持失败回滚
4. 侵入性尽量小，不影响现有功能

---

## 二、当前架构分析

### 2.1 核心组件差异

| 模式 | 专属组件 | 缓存策略 | 事件循环 | Bootstrap Server |
|------|----------|----------|----------|------------------|
| **NULL** | 无 | Radix Cache (可选) | `event_loop_normal` | 不需要 |
| **PREFILL** | `PrefillBootstrapQueue`<br>`disagg_prefill_inflight_queue`<br>`KVManager(PREFILL)` | Radix Cache (可用) | `event_loop_normal_disagg_prefill` | **需要启动** |
| **DECODE** | `DecodePreallocQueue`<br>`DecodeTransferQueue`<br>`KVManager(DECODE)` | Chunk Cache (强制禁用 Radix) | `event_loop_normal_disagg_decode` | 不需要 |

### 2.2 主要挑战

1. **组件初始化差异**: 不同模式初始化不同的 Queue 和 KVManager
2. **缓存策略不兼容**: DECODE 模式强制禁用 Radix Cache
3. **事件循环静态绑定**: 事件循环在启动时确定，运行在无限 `while True` 中
4. **Bootstrap Server**: 仅在 PREFILL 模式需要启动
5. **请求处理差异**: 不同模式对请求的处理流程完全不同

### 2.3 关键代码位置

- **Scheduler 初始化**: `python/sglang/srt/managers/scheduler.py` - `init_disaggregation()`
- **事件循环派发**: `python/sglang/srt/managers/scheduler.py` - `run_scheduler_process()` (lines 2935-2960)
- **PREFILL 组件**: `python/sglang/srt/disaggregation/prefill.py`
- **DECODE 组件**: `python/sglang/srt/disaggregation/decode.py`
- **Bootstrap Service**: `python/sglang/srt/managers/disagg_service.py`
- **参数处理**: `python/sglang/srt/server_args.py` - `_handle_pd_disaggregation()`

---

## 三、设计方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Unified Event Loop                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Mode Controller                             │  │
│  │  - current_mode: DisaggregationMode                           │  │
│  │  - target_mode: DisaggregationMode (for transition)           │  │
│  │  - transition_state: IDLE / CHECKING / SWITCHING / ROLLBACK   │  │
│  │  - snapshot: 切换前的状态快照 (用于回滚)                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│          ┌──────────────────┼──────────────────┐                    │
│          ▼                  ▼                  ▼                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  NULL Handler │  │PREFILL Handler│  │ DECODE Handler│           │
│  │   (Lazy Init) │  │  (Lazy Init)  │  │  (Lazy Init)  │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心设计原则

1. **先检查后切换**: 收到切换请求时，先检查队列是否为空，为空才执行切换
2. **原子切换**: 停止接受新请求 → 执行切换 → 恢复接受请求
3. **失败回滚**: 切换过程中任何步骤失败，自动回滚到原状态
4. **延迟初始化**: 组件按需初始化，仅在首次需要时创建
5. **状态快照**: 切换前保存状态快照，用于回滚

---

## 四、状态机设计

### 4.1 状态定义

```python
class TransitionState(Enum):
    IDLE = "idle"              # 正常运行状态，可接受新请求
    CHECKING = "checking"      # 检查是否可以切换（队列是否为空）
    SWITCHING = "switching"    # 正在执行切换，拒绝新请求
    ROLLBACK = "rollback"      # 切换失败，正在回滚
```

### 4.2 状态转换图

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
              ┌──────────┐                                │
              │   IDLE   │◄───────────────────────────────┤
              └────┬─────┘                                │
                   │                                      │
                   │ 收到切换请求                           │
                   ▼                                      │
              ┌──────────┐    队列非空                     │
              │ CHECKING │─────────────────────────────────┘
              └────┬─────┘    (拒绝切换请求)
                   │
                   │ 队列为空
                   ▼
              ┌──────────┐    切换成功      ┌──────────┐
              │SWITCHING │─────────────────►│   IDLE   │
              └────┬─────┘                  └──────────┘
                   │
                   │ 切换失败
                   ▼
              ┌──────────┐    回滚完成      ┌──────────┐
              │ ROLLBACK │─────────────────►│   IDLE   │
              └──────────┘                  └──────────┘
```

---

## 五、切换流程

### 5.1 成功切换流程

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. 收到切换请求: POST /admin/switch_disaggregation_mode            │
│     - 目标模式: target_mode                                          │
│     - 状态: IDLE → CHECKING                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. 检查队列状态                                                     │
│     - waiting_queue 是否为空                                         │
│     - running_batch 是否为空                                         │
│     - inflight_queue 是否为空 (PREFILL 模式)                         │
│     - prealloc_queue 是否为空 (DECODE 模式)                          │
│                                                                      │
│     如果非空: 状态 → IDLE，返回错误（队列非空，拒绝切换）                 │
│     如果为空: 继续下一步                                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. 开始切换 (状态: CHECKING → SWITCHING)                            │
│     a. 停止接受新请求                                                 │
│     b. 创建状态快照 (用于回滚)                                         │
│        - 当前模式                                                     │
│        - 缓存状态                                                     │
│        - 组件引用                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. 执行模式切换                                                     │
│     a. 如需切换缓存策略 (→DECODE 需禁用 Radix Cache)                  │
│     b. 初始化目标模式组件 (延迟初始化)                                 │
│        - PREFILL: PrefillBootstrapQueue, KVManager(PREFILL)         │
│        - DECODE: DecodePreallocQueue, DecodeTransferQueue           │
│     c. 如需启动/停止 Bootstrap Server                                 │
│        - → PREFILL: 启动 Bootstrap Server                            │
│        - PREFILL →: 停止 Bootstrap Server                            │
│     d. 更新 current_mode                                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. 切换完成 (状态: SWITCHING → IDLE)                                │
│     a. 恢复接受新请求                                                 │
│     b. 清除状态快照                                                   │
│     c. 返回成功响应                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 失败回滚流程

```
┌─────────────────────────────────────────────────────────────────────┐
│  切换过程中发生错误 (状态: SWITCHING → ROLLBACK)                      │
│                                                                      │
│  可能的失败原因:                                                      │
│  - 组件初始化失败 (如 KVManager 创建失败)                              │
│  - Bootstrap Server 启动失败 (端口占用等)                              │
│  - 缓存策略切换失败                                                   │
│  - 资源不足                                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  回滚操作                                                            │
│  a. 从快照恢复 current_mode                                          │
│  b. 销毁已创建的新组件                                                │
│  c. 恢复缓存策略                                                      │
│  d. 恢复 Bootstrap Server 状态                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  回滚完成 (状态: ROLLBACK → IDLE)                                    │
│  a. 恢复接受新请求                                                   │
│  b. 返回失败响应（包含错误原因）                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、核心类设计

### 6.1 ModeController 类

**文件位置**: `python/sglang/srt/disaggregation/mode_controller.py`

```python
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from sglang.srt.disaggregation.utils import DisaggregationMode

logger = logging.getLogger(__name__)


class TransitionState(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    SWITCHING = "switching"
    ROLLBACK = "rollback"


@dataclass
class ModeSnapshot:
    """切换前的状态快照，用于回滚"""
    mode: DisaggregationMode
    radix_cache_enabled: bool
    bootstrap_server_running: bool
    components: Dict[str, Any] = field(default_factory=dict)


class ModeTransitionError(Exception):
    """模式切换异常"""
    def __init__(self, message: str, can_retry: bool = False):
        super().__init__(message)
        self.can_retry = can_retry


class ModeController:
    """
    Disaggregation 模式控制器

    负责管理模式切换的完整生命周期，包括：
    - 状态检查
    - 组件初始化/销毁
    - 失败回滚
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.current_mode = DisaggregationMode(
            scheduler.server_args.disaggregation_mode
        )
        self.target_mode: Optional[DisaggregationMode] = None
        self.transition_state = TransitionState.IDLE
        self.snapshot: Optional[ModeSnapshot] = None
        self.last_error: Optional[str] = None

        # 组件懒加载标记
        self._prefill_components_initialized = (
            self.current_mode == DisaggregationMode.PREFILL
        )
        self._decode_components_initialized = (
            self.current_mode == DisaggregationMode.DECODE
        )

        # Bootstrap Server 管理器
        self._bootstrap_server = None

    def request_mode_switch(
        self,
        target_mode: DisaggregationMode
    ) -> tuple[bool, str]:
        """
        请求切换模式

        Args:
            target_mode: 目标模式

        Returns:
            (success, message): 是否成功接受请求，以及消息
        """
        # 检查当前状态
        if self.transition_state != TransitionState.IDLE:
            return False, f"切换正在进行中，当前状态: {self.transition_state.value}"

        if self.current_mode == target_mode:
            return True, f"已经处于 {target_mode.value} 模式"

        # 进入检查状态
        self.transition_state = TransitionState.CHECKING
        self.target_mode = target_mode
        self.last_error = None

        # 检查队列是否为空
        if not self._all_queues_empty():
            self.transition_state = TransitionState.IDLE
            self.target_mode = None
            return False, "队列非空，请等待所有请求处理完成后重试"

        # 队列为空，开始切换
        try:
            self._execute_switch()
            return True, f"成功切换到 {target_mode.value} 模式"
        except ModeTransitionError as e:
            self.last_error = str(e)
            return False, f"切换失败: {e}"
        except Exception as e:
            self.last_error = str(e)
            return False, f"切换失败（未知错误）: {e}"

    def can_accept_new_requests(self) -> bool:
        """是否可以接受新请求"""
        return self.transition_state == TransitionState.IDLE

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "current_mode": self.current_mode.value,
            "transition_state": self.transition_state.value,
            "target_mode": self.target_mode.value if self.target_mode else None,
            "last_error": self.last_error,
            "prefill_initialized": self._prefill_components_initialized,
            "decode_initialized": self._decode_components_initialized,
        }

    def _all_queues_empty(self) -> bool:
        """检查所有请求队列是否为空"""
        s = self.scheduler

        # 通用队列
        if len(s.waiting_queue) > 0:
            return False
        if not s.running_batch.is_empty():
            return False

        # PREFILL 模式特有队列
        if hasattr(s, 'disagg_prefill_bootstrap_queue'):
            if len(s.disagg_prefill_bootstrap_queue.queue) > 0:
                return False
        if hasattr(s, 'disagg_prefill_inflight_queue'):
            if len(s.disagg_prefill_inflight_queue) > 0:
                return False

        # DECODE 模式特有队列
        if hasattr(s, 'disagg_decode_prealloc_queue'):
            if not s.disagg_decode_prealloc_queue.is_empty():
                return False
        if hasattr(s, 'disagg_decode_transfer_queue'):
            if not s.disagg_decode_transfer_queue.is_empty():
                return False

        return True

    def _execute_switch(self):
        """执行模式切换"""
        self.transition_state = TransitionState.SWITCHING

        # 1. 创建状态快照
        self._create_snapshot()

        try:
            # 2. 处理缓存策略
            self._handle_cache_strategy()

            # 3. 初始化目标模式组件
            self._init_target_components()

            # 4. 管理 Bootstrap Server
            self._manage_bootstrap_server()

            # 5. 更新当前模式
            self.current_mode = self.target_mode
            self.scheduler.disaggregation_mode = self.target_mode

            # 6. 清理
            self.target_mode = None
            self.snapshot = None
            self.transition_state = TransitionState.IDLE

            logger.info(f"成功切换到 {self.current_mode.value} 模式")

        except Exception as e:
            logger.error(f"模式切换失败: {e}，开始回滚")
            self._rollback()
            raise ModeTransitionError(str(e))

    def _create_snapshot(self):
        """创建状态快照"""
        self.snapshot = ModeSnapshot(
            mode=self.current_mode,
            radix_cache_enabled=not self.scheduler.server_args.disable_radix_cache,
            bootstrap_server_running=self._bootstrap_server is not None,
            components={
                "prefill_initialized": self._prefill_components_initialized,
                "decode_initialized": self._decode_components_initialized,
            }
        )

    def _handle_cache_strategy(self):
        """处理缓存策略切换"""
        # DECODE 模式需要禁用 Radix Cache
        if self.target_mode == DisaggregationMode.DECODE:
            if not self.scheduler.server_args.disable_radix_cache:
                # 检查 Radix Cache 是否为空
                if not self.scheduler.tree_cache.is_empty():
                    raise ModeTransitionError(
                        "Radix Cache 非空，无法切换到 DECODE 模式。"
                        "请等待缓存清空或手动清除。",
                        can_retry=True
                    )
                # 禁用 Radix Cache
                self.scheduler.server_args.disable_radix_cache = True
                logger.info("已禁用 Radix Cache 以切换到 DECODE 模式")

        # 从 DECODE 切换到其他模式可以恢复 Radix Cache
        elif self.current_mode == DisaggregationMode.DECODE:
            # 可选：恢复 Radix Cache（根据原始配置）
            pass

    def _init_target_components(self):
        """初始化目标模式组件"""
        if self.target_mode == DisaggregationMode.PREFILL:
            if not self._prefill_components_initialized:
                self.scheduler._init_prefill_disaggregation()
                self._prefill_components_initialized = True
                logger.info("PREFILL 模式组件初始化完成")

        elif self.target_mode == DisaggregationMode.DECODE:
            if not self._decode_components_initialized:
                self.scheduler._init_decode_disaggregation()
                self._decode_components_initialized = True
                logger.info("DECODE 模式组件初始化完成")

    def _manage_bootstrap_server(self):
        """管理 Bootstrap Server"""
        from sglang.srt.managers.disagg_service import start_disagg_service

        if self.target_mode == DisaggregationMode.PREFILL:
            # 启动 Bootstrap Server
            if self._bootstrap_server is None:
                try:
                    self._bootstrap_server = start_disagg_service(
                        self.scheduler.server_args
                    )
                    logger.info("Bootstrap Server 启动成功")
                except Exception as e:
                    raise ModeTransitionError(f"Bootstrap Server 启动失败: {e}")

        elif self.current_mode == DisaggregationMode.PREFILL:
            # 停止 Bootstrap Server
            if self._bootstrap_server is not None:
                try:
                    self._bootstrap_server.shutdown()
                    self._bootstrap_server = None
                    logger.info("Bootstrap Server 已停止")
                except Exception as e:
                    logger.warning(f"Bootstrap Server 停止失败: {e}")

    def _rollback(self):
        """回滚到切换前状态"""
        self.transition_state = TransitionState.ROLLBACK

        if self.snapshot is None:
            logger.error("无法回滚：快照不存在")
            self.transition_state = TransitionState.IDLE
            return

        try:
            # 恢复缓存策略
            self.scheduler.server_args.disable_radix_cache = (
                not self.snapshot.radix_cache_enabled
            )

            # 恢复 Bootstrap Server 状态
            if self.snapshot.bootstrap_server_running:
                if self._bootstrap_server is None:
                    from sglang.srt.managers.disagg_service import start_disagg_service
                    self._bootstrap_server = start_disagg_service(
                        self.scheduler.server_args
                    )
            else:
                if self._bootstrap_server is not None:
                    self._bootstrap_server.shutdown()
                    self._bootstrap_server = None

            # 恢复组件状态标记
            self._prefill_components_initialized = (
                self.snapshot.components.get("prefill_initialized", False)
            )
            self._decode_components_initialized = (
                self.snapshot.components.get("decode_initialized", False)
            )

            logger.info(f"回滚成功，恢复到 {self.snapshot.mode.value} 模式")

        except Exception as e:
            logger.error(f"回滚失败: {e}")
        finally:
            self.target_mode = None
            self.snapshot = None
            self.transition_state = TransitionState.IDLE
```

### 6.2 Scheduler 修改

**文件位置**: `python/sglang/srt/managers/scheduler.py`

#### 6.2.1 添加统一事件循环

```python
@DynamicGradMode()
def event_loop_unified(self):
    """
    统一事件循环，支持动态模式切换

    根据 mode_controller.current_mode 动态选择执行逻辑
    """
    while True:
        # 1. 接收请求
        recv_reqs = self.recv_requests()

        # 2. 检查是否可以接受新请求
        if self.mode_controller.can_accept_new_requests():
            self.process_input_requests(recv_reqs)
        else:
            # 正在切换中，返回 503 错误或暂存
            self._reject_requests_during_switch(recv_reqs)

        # 3. 根据当前模式执行对应逻辑
        current_mode = self.mode_controller.current_mode

        if current_mode == DisaggregationMode.NULL:
            self._loop_iteration_normal()
        elif current_mode == DisaggregationMode.PREFILL:
            self._loop_iteration_prefill()
        elif current_mode == DisaggregationMode.DECODE:
            self._loop_iteration_decode()


def _loop_iteration_normal(self):
    """NULL 模式单次迭代"""
    if self._engine_paused:
        return

    batch = self.get_next_batch_to_run()
    self.cur_batch = batch

    if batch:
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)
    else:
        self.self_check_during_idle()

    self.last_batch = batch


def _loop_iteration_prefill(self):
    """PREFILL 模式单次迭代"""
    # 处理 bootstrap 完成的请求
    self.waiting_queue.extend(
        self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
    )

    # 获取下一个 batch
    batch = self.get_next_disagg_prefill_batch_to_run()
    self.cur_batch = batch

    if batch:
        result = self.run_batch(batch)
        self.process_batch_result_disagg_prefill(batch, result)
    else:
        self.self_check_during_idle()

    # 处理传输中的请求
    self.process_disagg_prefill_inflight_queue()

    self.last_batch = batch


def _loop_iteration_decode(self):
    """DECODE 模式单次迭代"""
    # 处理预分配队列
    self.disagg_decode_prealloc_queue.process()

    # 处理传输队列
    self.disagg_decode_transfer_queue.process()

    # 获取下一个 batch
    batch = self.get_next_disagg_decode_batch_to_run()
    self.cur_batch = batch

    if batch:
        result = self.run_batch(batch)
        self.process_batch_result_disagg_decode(batch, result)
    else:
        self.self_check_during_idle()

    self.last_batch = batch


def _reject_requests_during_switch(self, recv_reqs):
    """切换过程中拒绝新请求"""
    for req in recv_reqs:
        if hasattr(req, 'rid'):
            # 返回 503 Service Unavailable
            self._send_error_response(
                req.rid,
                "服务正在进行模式切换，请稍后重试",
                status_code=503
            )
```

#### 6.2.2 拆分初始化方法

```python
def _init_prefill_disaggregation(self):
    """
    PREFILL 模式组件初始化（支持延迟调用）

    初始化组件:
    - PrefillBootstrapQueue
    - disagg_prefill_inflight_queue
    - KVManager (PREFILL)
    - MetadataBuffers
    """
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.disaggregation.utils import (
        MetadataBuffers,
        ReqToMetadataIdxAllocator,
        TransferBackend,
    )

    # 初始化 metadata buffers
    if not hasattr(self, 'disagg_metadata_buffers'):
        self.disagg_metadata_buffers = MetadataBuffers(
            size=self.max_running_requests,
            hidden_size=self.tp_worker.model_runner.model_config.hidden_size,
            hidden_states_dtype=self.dtype,
        )
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
            size=self.max_running_requests
        )

    # 初始化 PrefillBootstrapQueue
    draft_token_to_kv_pool = getattr(
        self.tp_worker.model_runner, "draft_token_to_kv_pool", None
    )

    self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
        token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
        metadata_buffers=self.disagg_metadata_buffers,
        tp_rank=self.tp_rank,
        tp_size=self.tp_size,
        gpu_id=self.gpu_id,
        bootstrap_port=self.server_args.disaggregation_bootstrap_port,
        gloo_group=self.attn_tp_cpu_group,
        max_total_num_tokens=self.max_total_num_tokens,
        decode_tp_size=self.server_args.disaggregation_decode_tp,
        decode_dp_size=self.server_args.disaggregation_decode_dp,
        scheduler=self,
        pp_rank=self.pp_rank,
        pp_size=self.pp_size,
        transfer_backend=self.transfer_backend,
    )

    # 初始化 inflight queue
    self.disagg_prefill_inflight_queue = []

    logger.info("PREFILL disaggregation 组件初始化完成")


def _init_decode_disaggregation(self):
    """
    DECODE 模式组件初始化（支持延迟调用）

    初始化组件:
    - DecodePreallocQueue
    - DecodeTransferQueue
    - KVManager (DECODE)
    """
    from sglang.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeTransferQueue,
    )

    # 初始化 DecodeTransferQueue
    self.disagg_decode_transfer_queue = DecodeTransferQueue(
        scheduler=self,
    )

    # 初始化 DecodePreallocQueue
    self.disagg_decode_prealloc_queue = DecodePreallocQueue(
        scheduler=self,
        transfer_queue=self.disagg_decode_transfer_queue,
    )

    logger.info("DECODE disaggregation 组件初始化完成")
```

### 6.3 HTTP API 设计

**文件位置**: `python/sglang/srt/entrypoints/http_server.py`

```python
@app.post("/admin/switch_disaggregation_mode")
async def switch_disaggregation_mode(request: Request):
    """
    动态切换 disaggregation 模式

    Request Body:
    {
        "mode": "null" | "prefill" | "decode"
    }

    Response:
    - 200: 切换成功
    - 400: 无效的模式
    - 409: 切换正在进行中
    - 503: 队列非空，无法切换
    """
    data = await request.json()
    target_mode_str = data.get("mode")

    # 验证模式
    if target_mode_str not in ["null", "prefill", "decode"]:
        return JSONResponse(
            {"error": f"无效的模式: {target_mode_str}"},
            status_code=400
        )

    target_mode = DisaggregationMode(target_mode_str)

    # 请求切换
    success, message = await asyncio.to_thread(
        _global_state.scheduler.mode_controller.request_mode_switch,
        target_mode
    )

    if success:
        return JSONResponse({
            "status": "success",
            "message": message,
            "current_mode": target_mode_str
        })
    else:
        # 根据错误类型返回不同的状态码
        if "正在进行中" in message:
            status_code = 409
        elif "队列非空" in message:
            status_code = 503
        else:
            status_code = 500

        return JSONResponse(
            {"error": message},
            status_code=status_code
        )


@app.get("/admin/disaggregation_status")
async def get_disaggregation_status():
    """
    获取当前 disaggregation 状态

    Response:
    {
        "current_mode": "null" | "prefill" | "decode",
        "transition_state": "idle" | "checking" | "switching" | "rollback",
        "target_mode": null | "null" | "prefill" | "decode",
        "last_error": null | string,
        "prefill_initialized": bool,
        "decode_initialized": bool
    }
    """
    status = _global_state.scheduler.mode_controller.get_status()
    return JSONResponse(status)
```

---

## 七、模式切换矩阵

### 7.1 切换兼容性

| 源模式 → 目标模式 | 是否支持 | 特殊处理 |
|------------------|----------|----------|
| NULL → PREFILL | ✅ 支持 | 启动 Bootstrap Server |
| NULL → DECODE | ✅ 支持 | 禁用 Radix Cache，等待缓存清空 |
| PREFILL → NULL | ✅ 支持 | 停止 Bootstrap Server |
| PREFILL → DECODE | ✅ 支持 | 停止 Bootstrap Server，禁用 Radix Cache |
| DECODE → NULL | ✅ 支持 | 可选恢复 Radix Cache |
| DECODE → PREFILL | ✅ 支持 | 启动 Bootstrap Server，可选恢复 Radix Cache |

### 7.2 切换前置条件

| 条件 | 说明 |
|------|------|
| 队列为空 | waiting_queue, running_batch, inflight_queue 等必须为空 |
| → DECODE | Radix Cache 必须为空（如已启用） |
| → PREFILL | Bootstrap 端口可用 |

---

## 八、文件修改清单

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `srt/disaggregation/mode_controller.py` | **新建** | 模式控制器核心逻辑 |
| `srt/managers/scheduler.py` | 修改 | 添加统一事件循环、拆分初始化、集成 ModeController |
| `srt/entrypoints/http_server.py` | 修改 | 添加管理 API 端点 |
| `srt/server_args.py` | 修改 | 添加 `--enable-dynamic-disaggregation` 参数 |
| `srt/disaggregation/prefill.py` | 小改 | 适配延迟初始化 |
| `srt/disaggregation/decode.py` | 小改 | 适配延迟初始化 |
| `srt/managers/disagg_service.py` | 修改 | Bootstrap Server 支持动态启停 |

---

## 九、使用示例

### 9.1 启动支持动态切换的服务

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-dynamic-disaggregation \
  --disaggregation-mode null \
  --port 30000
```

### 9.2 切换到 PREFILL 模式

```bash
curl -X POST http://localhost:30000/admin/switch_disaggregation_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "prefill"}'
```

响应:
```json
{
  "status": "success",
  "message": "成功切换到 prefill 模式",
  "current_mode": "prefill"
}
```

### 9.3 查询当前状态

```bash
curl http://localhost:30000/admin/disaggregation_status
```

响应:
```json
{
  "current_mode": "prefill",
  "transition_state": "idle",
  "target_mode": null,
  "last_error": null,
  "prefill_initialized": true,
  "decode_initialized": false
}
```

### 9.4 队列非空时切换（失败案例）

```bash
curl -X POST http://localhost:30000/admin/switch_disaggregation_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "decode"}'
```

响应 (HTTP 503):
```json
{
  "error": "队列非空，请等待所有请求处理完成后重试"
}
```

---

## 十、测试计划

### 10.1 单元测试

1. **ModeController 状态机测试**
   - 状态转换正确性
   - 队列检查逻辑
   - 快照创建与恢复

2. **组件初始化测试**
   - 延迟初始化正确性
   - 重复初始化幂等性

### 10.2 集成测试

1. **基本切换测试**
   - NULL → PREFILL → DECODE → NULL 完整循环

2. **并发请求测试**
   - 切换过程中的请求处理
   - 队列非空时的拒绝逻辑

3. **失败回滚测试**
   - Bootstrap Server 启动失败
   - 组件初始化失败
   - 缓存策略切换失败

### 10.3 压力测试

1. **频繁切换测试**
2. **高并发请求下的切换**

---

## 十一、注意事项与限制

1. **TP/DP/PP 配置不可动态更改**: 这些配置涉及进程间通信和模型分片，无法在运行时更改

2. **切换期间服务短暂不可用**: 从检查队列到切换完成期间，新请求会被拒绝（返回 503）

3. **DECODE 模式缓存限制**: 切换到 DECODE 模式前必须等待 Radix Cache 清空

4. **Bootstrap Server 端口**: 动态启动时需确保端口未被占用

5. **内存占用**: 组件延迟初始化后会保留在内存中，不会自动释放

---

## 十二、后续优化方向

1. **请求暂存而非拒绝**: 切换期间将新请求暂存，切换完成后继续处理

2. **渐进式切换**: 支持等待队列自然清空，而非要求立即为空

3. **组件清理**: 支持释放不再使用的模式组件，减少内存占用

4. **切换预热**: 提前初始化目标模式组件，减少切换延迟
