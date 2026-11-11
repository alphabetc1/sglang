# Prefill scheduling & batch assembly: CPU-side optimization opportunities

Author: claude-sched (prefill 分析组). Scope: scheduler-side prefill path only
(`managers/scheduler.py`, `schedule_policy.py`, `schedule_batch.py`, `mem_cache/allocation.py`).
Method: read-only code analysis at commit 18107e38d2. **No measurements were taken**; every
"expected impact" below is a qualitative estimate to be confirmed by profiling
(e.g. py-spy on the scheduler process with a prefill-heavy workload).

Call chain under analysis:

```
Scheduler.get_next_batch_to_run            scheduler.py:3012
 └ get_new_batch_prefill                   scheduler.py:3154
   └ _get_new_batch_prefill_raw            scheduler.py:3180
     ├ SchedulePolicy.calc_priority        schedule_policy.py:237   (radix match per waiting req)
     ├ for req in waiting_queue:
     │   req.init_next_round_input(tree)   schedule_batch.py:1297   (radix match again)
     │   PrefillAdder.add_one_req          schedule_policy.py:1201
     ├ ScheduleBatch.init_new
     └ ScheduleBatch.prepare_for_extend    schedule_batch.py:2363
         └ alloc_for_extend                mem_cache/allocation.py:303
```

---

## 1. Radix prefix matching is done twice per admitted request, and the LPM pass copies the whole token list for every waiting request (highest priority)

**Evidence**

`schedule_policy.py:314` `SchedulePolicy._compute_prefix_matches` (runs on every prefill scheduling
pass when the policy is LPM and the queue holds 128 requests or fewer):

```python
for r in waiting_queue:
    prefix_ids = r.origin_input_ids + r.output_ids          # O(context) copy per req per pass
    match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids, include_req=True)
    ...
    if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
        match_result = self.waiting_queue_radix_tree.match_prefix(...)   # 2nd tree walk
        ...
        self.waiting_queue_radix_tree.insert(InsertParams(
            key=RadixKey(token_ids=prefix_ids, ...),
            value=torch.empty(len(prefix_ids), dtype=torch.bool)))       # tensor alloc per req
```

Then, in `scheduler.py:3323` `_get_new_batch_prefill_raw`, each request being admitted calls
`req.init_next_round_input(self.tree_cache)`, which goes into `schedule_batch.py:1358` and runs
`tree_cache.match_prefix(...)` **again** on the same tokens.

On top of that, `init_next_round_input` already avoids the copy: it uses the incrementally
maintained `full_untruncated_fill_ids` (`_refresh_fill_ids`, `schedule_batch.py:1276`), plus a
`limit=` cap. A comment at `schedule_batch.py:1326` says this is done to avoid "an O(context) copy per
prefill-batch build". `_compute_prefix_matches` and `match_prefix_for_req` (`schedule_policy.py:146`)
do not use that path and still build `origin_input_ids + output_ids`.

Also note `_determine_active_policy` (`schedule_policy.py:290`) **silently switches LPM to FCFS when the
queue exceeds 128 requests**. That is a throughput cliff: the cache-aware ordering turns off exactly
when the queue is long and its value is highest. It is a workaround for the cost above.

**Why it is slow**: for N waiting requests with average context L, each pass does about 2N tree walks
plus N list copies of length L plus up to N `torch.empty(L)` allocations, all in Python on the
scheduler thread. Long-context or agentic workloads (large L, many requests sharing prefixes) are the
worst case. The tree is unchanged between the two matches unless something was inserted or evicted in
between.

**Opportunities**
- In `_compute_prefix_matches`, call `r._refresh_fill_ids()` and pass `r.full_untruncated_fill_ids`
  instead of building `origin_input_ids + output_ids`. Do the same in the default branch of
  `match_prefix_for_req`.
- Cache the match result on the req with a tree "version/epoch" counter (bump it on
  insert/evict/split). `init_next_round_input` could then skip the second `match_prefix` when the epoch
  is unchanged and only the `limit` is tighter. Mamba COW and hicache side effects need care: keep the
  second match only when `cow_mamba` or a host hit is involved.
- The in-batch simulated tree only needs a key, not a value. Use a shared zero-length or sentinel
  value instead of `torch.empty(len(prefix_ids))`.
- Once the per-pass cost drops, the hard-coded `128` threshold could become configurable or adaptive,
  or be replaced by incremental re-sorting.

**Expected impact**: lower scheduler CPU time per prefill pass, which grows with queue length ×
context length. It also lets LPM stay active for larger queues, which may improve the cache hit rate.
Not measured.

---

## 2. `prepare_for_extend` / `alloc_for_extend` do synchronous or pageable H2D copies and per-req small device ops

**Evidence**

`mem_cache/allocation.py:303` `alloc_for_extend`:

```python
prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)       # pageable
extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)       # pageable
prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)   # non_blocking ineffective on pageable mem
extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)
req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)   # pageable
req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)
...
last_loc = [ (t[-1:] if len(t) > 0 else torch.full((1,), -1, device=batch.device))
             for t in prefix_tensors ]                                       # B tiny device ops
... last_loc=torch.cat(last_loc)
```

`schedule_batch.py:2363` `prepare_for_extend`:

```python
seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(self.device, non_blocking=True)
seq_lens_cpu    = torch.tensor(seq_lens, dtype=torch.int64)       # same list converted twice
...
if mamba_extra_buffer_enabled():
    self.mamba_track_indices = torch.tensor(mamba_track_indices_cpu, dtype=torch.int64, device=self.device)  # sync H2D x3
    self.mamba_track_mask    = torch.tensor(..., device=self.device)
    self.mamba_track_seqlens = torch.tensor(..., device=self.device)
...
replace_positions_tensor = torch.tensor(all_replace_positions, dtype=torch.long, device=self.device)  # sync H2D
```

`input_ids` already follows the right pattern: it is flattened into pinned CPU memory and the H2D copy
is deferred to the forward stream (`flatten_arrays_to_pinned_cpu`, comment "H2D is deferred to forward
stream via resolve_forward_inputs"). The other per-batch metadata does not.

**Why it is slow**: `.to(cuda, non_blocking=True)` from pageable memory goes through a staging buffer
and blocks the host. `torch.tensor(list, device=cuda)` is a blocking copy. With overlap scheduling, any
host-blocking copy on the scheduler thread eats into the CPU time hidden behind the previous GPU batch.
In the paged `last_loc` path, B per-request slice and `torch.full` launches plus a `cat` scale linearly
with batch size, which hurts many-short-prompt batches the most.

**Opportunities**
- Pack `prefix_lens`, `extend_lens`, `seq_lens`, `orig_seq_lens` and `req_pool_indices` into **one
  pinned int64 CPU buffer** and issue a single H2D copy. Slice views on the device. Reuse
  `seq_lens_cpu` as the pinned source instead of building it twice.
- Compute `last_loc` on the CPU side: the last prefix index per req is known from `prefix_indices`. Or
  use a single gather kernel over `req_to_token[req_pool_indices, prefix_lens-1]` instead of a per-req
  Python loop of device slices.
- Move the mamba track tensors and `replace_positions` to the same pinned/deferred path as `input_ids`.

**Expected impact**: fewer host-blocking syncs and kernel launches per prefill batch. The benefit is
largest for high-QPS short-prompt workloads where scheduler overhead is a visible share of a step.
Not measured.

---

## 3. `input_embeds` and logprob token ids are assembled as Python lists of floats/ints

**Evidence** (`schedule_batch.py:2437-2442` and `schedule_batch.py:2568`):

```python
if req.input_embeds is not None:
    input_embeds.extend(req.input_embeds[pre_len : pre_len + req.extend_range.length])   # List[List[float]]
...
self.input_embeds = torch.tensor(input_embeds, pin_memory=_pin).to(self.device, non_blocking=True)
```

`Req.input_embeds` is typed `Optional[List[List[float]]]` (`schedule_batch.py:828`). Converting
`tokens × hidden_size` Python floats with `torch.tensor` every prefill (and every chunk) costs
O(tokens·hidden) Python-object work. With hidden=4096 and an 8k-token prompt that is about 33M floats.

The logprob path (`schedule_batch.py:2528-2535`) is similar: `extend_input_logprob_token_ids.extend([0] * n)`
plus a non-pinned `torch.tensor(...)` built per batch.

**Opportunities**: convert `input_embeds` to a (pinned) `torch.Tensor` once, at request ingestion
(tokenizer manager / `Req.__init__`). Then slicing in `prepare_for_extend` becomes a zero-copy view and
`torch.cat`. For logprob ids, pre-size a numpy/torch buffer instead of growing a Python list.

**Expected impact**: large for the `input_embeds` API path and for chunked prefill of long embed
inputs; none for normal token-id requests. Not measured.

---

## Minor items
- `schedule_batch.py:1335-1339` and `1354-1357`: `swa_reprefill_tail_tokens()` is computed and applied
  twice in `init_next_round_input`. The second application is redundant (idempotent `min`).
- `scheduler.py:3370`: `self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]`
  rebuilds the queue on every pass (O(N)). Admitted requests are a prefix of the iteration order
  except for skipped LoRA or prefetch reqs, so a deque or index-based removal would suffice. Low priority.
- `schedule_policy.py:387` `_sort_by_dfs_weight` walks the **entire** radix tree recursively
  (`_calc_weight`, `_get_dfs_priority`) on every pass. Its cost scales with cache size, not queue
  size, and deep trees risk hitting the recursion limit. It could be limited to the subtree spanned by
  the waiting requests' `last_node`s.

## Suggested validation (not run here, read-only trial)
1. Run py-spy `record` on the scheduler process under a long-context shared-prefix workload with
   `--schedule-policy lpm`, and compare self-time in `_compute_prefix_matches`, `match_prefix` and
   `prepare_for_extend`.
2. Take an nsys trace of a high-QPS short-prompt workload and count `cudaMemcpy` (sync) calls and tiny
   kernels issued from `alloc_for_extend` per prefill step.
