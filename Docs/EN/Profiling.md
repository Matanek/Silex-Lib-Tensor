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

All events count work issued during the period. Live buffers and active bytes
instead describe the device's current state, including Tensor buffers created
before the period began.

| `ProfileStatistics` fields | Meaning |
| --- | --- |
| `uploads`, `downloads` | Tensor CPU→GPU and GPU→CPU transfers. |
| `uploaded_bytes`, `downloaded_bytes` | Logical transfer bytes according to dtype. |
| `buffer_allocations`, `buffer_releases` | Tensor storage buffer creation and release during the period. Releases may concern earlier buffers. |
| `live_buffers`, `peak_live_buffers` | Currently live Tensor buffers and maximum since the period began. |
| `active_bytes`, `peak_bytes` | Sum of requested sizes of these buffers and maximum during the period. |
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

A period follows calls from the calling thread; it provides no concurrent
collection guarantee. Inactive collection keeps the buffer tracking needed to
open a profile on already resident tensors. Active collection adds increments
for live periods at each event. Cost depends on the number of simultaneous
profiles and workload. `Neural.sx` distinguishes `--measure-mlp`/`--measure-cnn`/
`--measure-rnn` (no profile) from their `--profile-mlp`/`--profile-cnn`/
`--profile-rnn` variants. Snapshots and printing occur after timed sections.

Release qualification on 5 October 2026, Apple M3 Pro / Metal, retains 75
processes (five repetitions per variant/family and A/A controls). Across nine
complete resident steps, median changes with one active period range from
−3.11% to +5.35% compared to the candidate without a profile. Other phases have
positive changes up to +20.75%; A/A controls themselves show signals. This
proves neither zero cost nor universal non-regression. Raw data, MAD and
counterchecks are retained under
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/2026-10-05-macos-arm64-public-profile/`.
