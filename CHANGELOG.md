# Tensor release notes

## [Unreleased]

- Adam avoids recomputing zero moments after their first initialization on
  CPU and GPU. The formulas, moment values, and deferred initialization of
  parameters remain unchanged. Subsequent GPU steps remove two passes and
  two temporary allocations per active parameter.
