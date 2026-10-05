# Observe GPU costs

`Tensor.Profile` observes Tensor work issued on one device during a period. Use
it to distinguish resident computation, transfers and buffer pressure before
optimizing a training workload.

```sx
use GFX.GPU
use Tensor
use Tensor.Profile

var device = GPU.Device()
var values:float[] = [1.0, 2.0, 3.0]
let input = Tensor.vector(values).to(device)
var profile = Profile(device)
let output = input.multiply(2.0)
let resident = profile.snapshot()
assert(resident.uploads == 0 && resident.downloads == 0)
assert(resident.dispatches == 1 && resident.active_bytes == 24)

profile.reset()
let observed = output.cpu().values()
let observation = profile.snapshot()
assert(observation.downloads == 1 && observation.downloaded_bytes == 12)
assert(resident.downloads == 0)
```

`use Tensor.Profile` selects the profile type; `use Tensor.ProfileStatistics`
imports the snapshot type when an annotation is needed. `Profile` is a
non-clonable class. Its period starts at construction and ends at destruction.
`reset()` starts a new period on the same device. Multiple profiles may observe
overlapping periods; resetting or destroying one does not change any other.

`snapshot()` returns an immutable value containing only integer counters. Later
operations and resets leave earlier snapshots unchanged. Reading, resetting or
destroying a profile downloads no elements, creates no fence and waits for no
GPU command. To measure wall time including GPU completion, explicitly call
`device.wait_until_idle()` before and after the timed workload. A submission
counter does not measure GPU execution time.

## Counters

All events count work issued during the period. Buffer counts and bytes describe
the current device state, including Tensor reservations created before the period
started. Tensor can reuse ready buffers: distinguish live native resources from
buffers still in use by values.

| `ProfileStatistics` fields | Meaning |
| --- | --- |
| `uploads`, `downloads` | Tensor CPU→GPU and GPU→CPU transfers. |
| `uploaded_bytes`, `downloaded_bytes` | Logical transfer bytes according to dtype. |
| `buffer_allocations`, `buffer_releases` | Actual Tensor storage buffer creation and release during the period. Reuse creates no buffer. Releases may concern earlier buffers. |
| `buffer_reuses` | Ready buffer acquisitions without a new native creation. |
| `cache_evictions` | Buffers removed from the cache to respect capacity or clear it at the last storage. |
| `live_buffers`, `peak_live_buffers` | Live native Tensor buffers, including ready cached buffers, and maximum since the period began. |
| `in_use_buffers`, `peak_in_use_buffers` | Buffers owned by value storage and maximum during the period; views share their storage. |
| `active_bytes`, `peak_bytes` | Sum of requested sizes of buffers in use by values and maximum during the period. |
| `allocated_bytes`, `peak_allocated_bytes` | Sum of requested sizes of all live native buffers, including the cache, and maximum during the period. |
| `cached_buffers`, `cached_bytes` | Available ready buffers and sum of their requested sizes. |
| `cache_capacity_buffers`, `cache_capacity_bytes` | Per-device cache limits: currently 32 buffers and 1 MiB. |
| `upload_staging_allocations`, `download_staging_allocations` | Corresponding Tensor transfer buffer creations. |
| `fence_creations`, `submissions` | Fences created and command buffers submitted by Tensor. |
| `pipeline_compilations` | Tensor pipeline creations; cache reuse adds none. |
| `compute_passes`, `pipeline_bindings`, `dispatches` | Compute passes, pipeline bindings and dispatches actually issued by Tensor. |

`reset()` clears events and starts peaks from currently live buffers and active
bytes. Bytes describe requested storage buffer sizes, not physical VRAM: they
exclude staging, shaders and driver memory. A view shares a buffer and does not
count as an allocation. Empty storage currently reserves a minimum four-byte
buffer without transfer. GPU copy materialization can submit commands and create
a fence without a compute pass or dispatch.

## Attribution and lifetime

A profile counts only Tensor calls on the selected device. External GFX commands
and `device.reset_command_stats()` leave its counters unchanged. CPU computation
and other devices are excluded. An explicit GPU extraction through `cpu()` counts
as a download.

Storage retains the parents needed until commands complete, so its bytes remain
active while its buffer is retained. A profile retains no tensor: it keeps its
device and period data. Internal device state disappears when no Tensor storage
or profile remains alive; no global device or period history is retained.

## Reuse and reserved memory

Tensor's private cache only acquires a buffer whose fence is complete and on
which no live storage, view or branch depends. Size and dtype must match
exactly. Buffers larger than the limit are released directly. The cache respects
both 32 entries and 1 MiB per device; the oldest ready entry is removed when a
limit would be exceeded. The optimization reuses storage without grouping
commands or avoiding computation fences.

`live_buffers == in_use_buffers + cached_buffers` and
`allocated_bytes == active_bytes + cached_bytes` describe current state.
Maximums are observed separately and need not add up. After temporary
computation, `active_bytes` can return to its previous level while
`allocated_bytes` remains higher and `buffer_releases` is unchanged: the buffer
is available for a later computation. `reset()` does not clear the cache. It
starts all peaks from current state and only zeroes events; `snapshot()` changes
no reservation.

The cache is cleared when the device's last Tensor storage disappears, even if
a `Profile` remains alive. Destroying a profile does not clear the cache of
values still alive. Requested sizes include provider minimums but measure
neither physical Metal memory nor internal driver allocations. The cache adds
no transfer or wait to profiling.

The new `ProfileStatistics` fields default to zero, preserving named
constructions that only provide the earlier fields. Snapshots produced by
`Profile` report the effective limits.

A period follows calls from the calling thread; it provides no concurrent
collection guarantee. Inactive collection keeps the buffer tracking needed to
open a profile on already resident tensors. Active collection adds increments
for live periods at each event. Cost depends on the number of simultaneous
profiles and workload. `Neural.sx` distinguishes `--measure-mlp`/`--measure-cnn`/
`--measure-rnn` (no profile) from their `--profile-mlp`/`--profile-cnn`/
`--profile-rnn` variants. Snapshots and printing occur after timed sections.

The initial public-profile qualification, before buffer reuse, in Release on
5 October 2026, Apple M3 Pro / Metal, retains 75
processes (five repetitions per variant/family and A/A controls). Across nine
complete resident steps, median changes with one active period range from
−3.11% to +5.35% compared to the candidate without a profile. Other phases have
positive changes up to +20.75%; A/A controls themselves show signals. This
proves neither zero cost nor universal non-regression. Raw data, MAD and
counterchecks are retained under
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/2026-10-05-macos-arm64-public-profile/`.

## Measured reuse result

The Release comparison on 5 October 2026, Apple M3 Pro / Metal, uses the same
exact compiler, 16 eager/cache pairs and eight A/A controls for each binary.
Three resident Transformer steps change from 40.260 ms to 37.853 ms in median;
the median paired difference is −5.89%, with a MAD of 1.20 percentage points
and 14 favorable pairs out of 16. Controls are valid under the rule fixed
before measurement. All 1,700 values across the six models, including gradients
and states, remain qualified at tolerance 2e-5 with AdamW epsilon1e-4.

The five other models and twelve microcases demonstrate no gain under the
admission rule. Cold costs remain indeterminate. Median Transformer native
creations decrease from 1,365 to 643; passes and submissions remain identical.
Fewer creations do not mean all workloads run faster. Median process RSS for
the six-family workload increases from 81,510,400 to 83,869,696 bytes; it includes
the driver and allocator and does not measure physical GPU memory. The cache
keeps at most 1 MiB of requested buffer sizes per device.

Sources, configurations, raw data, dispersion, controls and a verifier belong
to `Silex-Benchmarks/Sources/TensorStableCompute/Baselines/` under
`2026-10-05-macos-arm64-ready-buffer-reuse-micro` and
`2026-10-05-macos-arm64-ready-buffer-reuse-models`. This result concerns these
workloads, hardware and binaries; no performance generalization to other
targets is established.

## Qualified normalization fusion

GPU float32 `Tensor.layer_norm` computes epsilon addition and square root in
one elementwise pass. One normalization emits six passes instead of seven,
avoiding one temporary, submission and fence. Mean and variance reductions
keep their order. The APIs and CPU computation preserve their behavior;
gradients, views, aliases and nonfinite values are qualified.

On October 6, 2026, Apple M3 Pro / Metal native Release measurements with
16 pairs and eight A/A controls per binary admit three gains: 9.50% for one
value, 10.38% for seven values and 9.08% for a transpose/narrow view of 49 values.
Their paired MADs are 0.87, 0.85 and 0.94 percentage points; all 16 pairs are
favorable in each case. Samples complete 512 normalizations after 16 warmups,
with epsilon 0.25. Sizes 257/4096/65536, all six model families and all cold
costs remain indeterminate. No regression is admitted; this does not prove
neutrality for those cases.

All 108,800 scalar observations across the six model families agree at the
fixed tolerance 2e-5 (AdamW epsilon 1e-4). Transformer avoids six passes over
three steps, with no admitted timing gain on this workload. Median model
process RSS changes from 86,392,832 to 87,441,408 bytes; microcase RSS changes
from 72,925,184 to 72,679,424 bytes. These measurements include the driver and
do not represent physical GPU memory.

The fusion also preserves the previously observed float32 GPU mean bias for
seven values of 1000 with epsilon 0.25: output −0.00012207031, identical to
eager computation. This witness is kept separately from the independent
scalar oracle; no tolerance is widened to hide it.

Recomputable campaigns belong to
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/` under
`2026-10-06-macos-arm64-admitted-normalization-fusion-micro` and
`2026-10-06-macos-arm64-admitted-normalization-fusion-models`.
Compilation and linking cover the six distributed targets; execution and
performance are established only on macOS arm64 / Metal.
