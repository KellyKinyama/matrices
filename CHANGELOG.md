## Unreleased

- Add statically-typed `scale()` to `Matrix64`, `Matrix32`, `Vector64`, and
  `Vector32`, and explicit axis broadcasting (`addVector`, `subtractVector`,
  `multiplyVector`, `divideVector`) on the matrix types to give a fully typed
  multiply path and remove the square-matrix broadcast ambiguity.
- Add a CPU reverse-mode automatic differentiation layer (`Variable`) built on
  the existing SIMD kernels, covering matmul, elementwise arithmetic,
  activations, reductions, broadcast-with-reduce bias, and memory-saving graph
  freeing in `backward`.
- Add a neural-network module: `Module`, `Linear`, `ReLU`, `Tanh`, `Sigmoid`,
  `Sequential`, and `SGD` (with momentum) / `Adam` optimizers.

## 2.0.0

- Redesign the package around high performance and script-friendly ergonomics,
  with Float32 matrix and vector types.
- Replace nested-list internals with contiguous typed-data storage and pure Dart
  SIMD kernels for dense matrix and vector hot paths.
- Add broad linear algebra coverage, including direct solvers, decompositions,
  eigen/SVD, pseudoinverse, least squares, PCA, iterative solvers, and Krylov
  methods.
- Add CSR sparse matrix support with dense conversion, sparse-vector products,
  sparse-dense multiplication, transpose, and JSON serialization.
- Align the 32-bit and 64-bit APIs across construction, arithmetic, statistics,
  serialization, solvers, decompositions, advanced algorithms, and sparse
  conversion.
- Add reproducible correctness, numerical robustness, performance regression,
  and benchmark-suite coverage.

## 1.2.7

- Documentation update.

## 1.2.6

- Bug fixes.

## 1.2.4

- Bug fixes.

## 1.2.3

- Bug fixes.

## 1.2.2

- Bug fixes.
- Add documentation webpage.

## 1.2.1

- Bug fixes.
- Add documentation webpage.

## 1.2.0

- Bug fixes.
- Major update; see the documentation.

## 1.1.5

- Bug fixes.
- Add `rowCount`, `columnCount`, and `count`.
- Add `max` and `min`.

## 1.1.4

- Bug fixes.
- Add `fromFlattenedList`.
- Add identity matrix support.

## 1.1.3

- Bug fixes.

## 1.1.2

- Bug fixes.
- Improve inverse.
- Standardize methods.

## 1.1.0

- Bug fixes.
- Remove emoji methods.
- Standardize methods.

## 1.0.11

- Bug fixes.
- Greatly improve determinant performance.

## 1.0.9

- Bug fixes.

## 1.0.8

- Update.

## 1.0.7

- Documentation update.

## 1.0.6

- Documentation update.

## 1.0.5

- Add English documentation.

## 1.0.4

- Documentation update.

## 1.0.3

- Bug fixes.

## 1.0.2

- Open source release.
