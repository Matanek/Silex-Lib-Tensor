# Diagnostic fixtures

Run each `.sx` file explicitly from the `SilexProject` root with `silex run`.
Every fixture must fail and mention its intended contract:

- `Extraction.sx`: matching dtype extractor or explicit cast;
- `GPUExtraction.sx`: `cpu()` before value extraction;
- `CastRange.sx`: invalid controlled numeric conversion;
- `GPUInteger.sx`: GPU scalar arithmetic accepts only `float32`;
- `CrossDevice.sx`: explicit CPU staging between devices;
- `Placement.sx`: matching placements;
- `DTypeMismatch.sx`: matching dtypes;
- `CopyTensor.sx`: a value reaching a `noncopyable` class cannot be cloned.
