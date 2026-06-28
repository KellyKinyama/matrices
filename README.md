# Matrices

[English](https://github.com/abandoft/matrices) | [中文文档](https://github.com/abandoft/matrices/blob/master/README-ZH.md)

Matrices is a matrix computation and linear algebra library for Dart and
Flutter. It focuses on high performance and script-style ergonomics, providing
NumPy/MATLAB-like construction, indexing, broadcasting, and operator semantics,
while keeping explicit 32-bit and 64-bit precision choices.

Project features:

- Uses Dart's built-in `SIMD` APIs to accelerate computation, without depending
  on FFI or platform-specific binary runtimes.
- Uses contiguous typed-data storage to avoid performance and semantic issues
  from nested-list structures.
- Explicit `Matrix64` / `Matrix32` and `Vector64` / `Vector32` APIs.
- Covers common dense matrices, CSR sparse matrices, direct decompositions,
  advanced decompositions, iterative methods, and Krylov methods.

## Quick Start

```dart
import 'package:matrices/matrices.dart';

void main() {
  final a = mat([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  final b = mat([
    [7, 8],
    [9, 10],
    [11, 12],
  ]);

  final x = vec([1, 1, 1]);

  print(a * b);      // Matrix64 matrix multiplication
  print(a * x);      // Matrix64 * Vector64
  print(a + 1);      // scalar broadcasting
  print(a + x);      // row-vector broadcasting
  print(a.transpose);
}
```

Explicit 32-bit path:

```dart
final a32 = mat32([
  [1, 2],
  [3, 4],
]);

final x32 = vec32([1, 1]);

print(a32 * x32);       // Vector32
print(a32 * a32);       // Matrix32
print(a32.toFloat64()); // Matrix64
```

## Type System

| Type | Storage | Use |
|---|---|---|
| `Matrix64` | `Float64List` | Default high-precision dense matrix |
| `Matrix32` | `Float32List` | Throughput and memory-oriented dense matrix |
| `Vector64` | `Float64List` | Default high-precision vector |
| `Vector32` | `Float32List` | Throughput-oriented float32 vector |
| `Matrix` | `typedef Matrix = Matrix64` | Script-friendly short name |
| `Vector` | `typedef Vector = Vector64` | Script-friendly short name |

`Matrix` and `Vector` are 64-bit aliases, not separate implementations. Public
APIs and documentation should prefer `Matrix64` / `Vector64` when precision must
be explicit. Use `Matrix32` / `Vector32` for float32. The package does not
silently downgrade precision.

Default constructors use full words, with short aliases for script-style code.
Examples use short aliases for interactive ergonomics.

| Default API | Short alias | Return type |
|---|---|---|
| `matrix([...])` | `mat([...])` | `Matrix64` |
| `matrix32([...])` | `mat32([...])` | `Matrix32` |
| `vector([...])` | `vec([...])` | `Vector64` |
| `vector32([...])` | `vec32([...])` | `Vector32` |

Helper constructors follow the same pattern, for example `vectorZeros` /
`vecZeros` and `vectorLinspace32` / `vecLinspace32`.

## Construction

64-bit matrices:

```dart
final a = Matrix64([
  [1, 2],
  [3, 4],
]);

final b = Matrix([
  [1, 2],
  [3, 4],
]); // Matrix is an alias for Matrix64

final z = zeros(2, 3);
final o = ones(2, 3);
final f = full(2, 3, 9);
final r = rand(2, 3, seed: 42);
final i = eye(4);
final d = diag([1, 2, 3]);
final grid = arange(0, 12, columns: 4);
final samples = linspace(0, 1, 5);
```

32-bit matrices:

```dart
final a32 = Matrix32([
  [1, 2],
  [3, 4],
]);

final z32 = zeros32(2, 3);
final o32 = ones32(2, 3);
final f32 = full32(2, 3, 1.5);
final r32 = rand32(1024, 1024, seed: 7);
final i32 = eye32(4);
final d32 = diag32([1, 2, 3]);
final grid32 = arange32(0, 12, columns: 4);
final samples32 = linspace32(0, 1, 5);
```

Other constructors:

```dart
final fromRows = Matrix64.fromRows([[1, 2], [3, 4]]);
final fromColumns = Matrix64.fromColumns([[1, 3], [2, 4]]);
final fromFlat = Matrix64.fromFlat([1, 2, 3, 4], 2, 2);
final fromBytes = Matrix64.fromByteData(
  Float64List.fromList([1, 2, 3, 4]).buffer.asByteData(),
  2,
  2,
);
final spd = Matrix64.randomSPD(4, seed: 1);
```

## Indexing, Views, And Conversion

```dart
final a = arange(0, 9, columns: 3);

print(a(1, 2));      // element access
print(a[1][2]);      // row view access
print(a.row(0));     // Vector64
print(a.column(1));  // Vector64

a.set(0, 0, 99);
a[1][1] = 42;

final rows = a.toRows();
final values = a.values;             // Float64List copy
final unsafe = a.unsafeValuesView;    // internal storage view for advanced use
final json = a.toJson();
final restored = Matrix64.fromJson(json);
```

Rows are fixed-length views. `Vector64` and `Vector32` are fixed-length linear
algebra objects that implement `Iterable<double>`; they are not growable
`List<double>` objects.

## Arithmetic Semantics

```dart
final a = mat([
  [1, 2],
  [3, 4],
]);
final b = mat([
  [5, 6],
  [7, 8],
]);
final x = vec([1, 1]);

print(a + b);          // elementwise addition
print(a - 1);          // scalar broadcasting
print(a * 2);          // scalar multiplication
print(a * b);          // matrix multiplication
print(a * x);          // matrix-vector multiplication
print(x * a);          // vector-matrix multiplication
print(a + x);          // row-vector broadcasting
print(a.hadamard(b));  // Hadamard elementwise multiplication
print(a / 2);
```

The meaning of `*` depends on the right-hand side:

- `matrix * matrix` is matrix multiplication.
- `matrix * vector` is matrix-vector multiplication.
- `vector * matrix` is row-vector matrix multiplication.
- `matrix * number` and `vector * number` are scalar multiplication.
- Use `hadamard` for elementwise matrix multiplication.

Invalid shapes throw `ArgumentError` immediately. The library does not silently
reshape, pad, or truncate data.

## Shape And Data Operations

```dart
final a = arange(0, 12, columns: 4);

print(a.transpose);
print(a.t);
print(a.reshape(4, 3));
print(a.slice(rowStart: 1, rowEnd: 3, columnStart: 1));
print(a.sample(rowIndices: [0, 2], columnIndices: [1, 3]));
print(a.vstack(a));
print(a.hstack(a));
print(a.flatten());
```

Transforms and statistics:

```dart
final centered = a.mapColumns((column) => column - column.mean);
final selected = a.filterRows((row, index) => row.sum > 10);
final sorted = a.sort((row) => row.sum, direction: SortDirection.desc);

print(a.mean);
print(a.meanByAxis(Axis.columns));
print(a.variance(Axis.rows));
print(a.deviation(Axis.rows));
print(a.norm());
```

## Vector API

```dart
final x = vec([1, 2, 3]);
final y = vec([4, 5, 6]);
final x32 = vec32([1, 2, 3, 4]);
final y32 = vec32([5, 6, 7, 8]);

print(x.dot(y));
print(x.norm());
print(x.distanceTo(y, Distance.euclidean));
print(x.cosine(y));
print(x.normalize());
print(x.subvector(1));
print(x.unique());

print(x32.dot(y32));
print(x32 + y32);
```

Helper constructors:

```dart
final z = vecZeros(3);
final o = vecOnes(3);
final f = vecFull(3, 2);
final r = randVec(3, seed: 1);
final range = vecRange(0, 10, step: 2);
final line = vecLinspace(0, 1, 5);

final z32 = vecZeros32(4);
final o32 = vecOnes32(4);
```

## Direct Linear Algebra

```dart
final a = mat([
  [4, 7],
  [2, 6],
]);

print(a.determinant);
print(a.inverse);
print(a.trace);
print(a.rank);
print(a.rref());

final rhs = mat([
  [1],
  [0],
]);
print(a.solve(rhs));
```

Square-matrix entry point:

```dart
final s = SquareMatrix.fromList([
  [4, 7],
  [2, 6],
]);

print(s.determinant);
print(s.inverse);
print(s.logAbsDeterminant);
```

Direct solve, determinant, and inverse use pivoting. Singular matrices throw at
working precision. Ill-conditioned problems should pass explicit tolerances.

## Decompositions And Advanced Algorithms

```dart
final a = mat([
  [4, 1],
  [1, 3],
]);

final lu = a.lu();
final qr = a.qr();
final cholesky = a.cholesky();
final eigen = a.eigenSymmetric();
final svd = a.svd();

print(lu.solve(mat([[1], [2]])));
print(qr.q * qr.r);
print(cholesky.lower * cholesky.lower.transpose);
print(eigen.values);
print(svd.singularValues);
```

High-level APIs:

```dart
final design = mat([
  [1, 1],
  [1, 2],
  [1, 3],
]);
final observed = mat([
  [1],
  [2],
  [2],
]);

print(design.leastSquares(observed));
print(design.pseudoInverse());

final pca = design.pca(components: 1);
print(pca.components);
print(pca.explainedVariance);
```

| API | Applies to | Typical use |
|---|---|---|
| `lu()` | square matrices | Direct solve, determinant, pivoted factorization |
| `qr()` | tall or full-rank square matrices | Least squares, orthogonalization |
| `cholesky()` | symmetric positive definite matrices | SPD solve and factorization |
| `eigenSymmetric()` | real symmetric matrices | Full eigenvalues and eigenvectors |
| `eigen()` | symmetric full solve or power iteration | High-level eigen interface |
| `svd()` | rectangular or square matrices | Low-rank analysis, pseudoinverse |
| `pseudoInverse()` | rectangular or square matrices | Moore-Penrose pseudoinverse |
| `leastSquares(rhs)` | overdetermined or full-rank systems | Least squares |
| `pca()` | observation x feature matrix | Principal component analysis |

`Matrix32` exposes the same high-level API names. Matrix-valued results keep
32-bit storage; scalar accumulations may use Dart `double` temporaries where the
language requires it.

## Iterative And Krylov Methods

```dart
final a = mat([
  [4, 1],
  [1, 3],
]);
final b = vec([1, 2]);

print(a.jacobi(b).solution);
print(a.gaussSeidel(b).solution);
print(a.sor(b, omega: 1.1).solution);
print(a.conjugateGradient(b).solution);
print(a.gmres(b, restart: 20).solution);
print(a.arnoldi(b, 2).hessenberg);
print(a.powerIteration().eigenvalue);
```

Result objects include convergence status, iteration count, residual norm, and
the computed solution or eigenpair. Iterative methods are sensitive to initial
guess, tolerance, conditioning, and expected convergence properties. Production
code should set `tolerance` and `maxIterations` explicitly.

## Sparse Matrices

```dart
final sparse = SparseMatrix.fromRows([
  [1, 0, 2],
  [0, 0, 3],
  [4, 0, 0],
]);

print(sparse.nnz);
print(sparse.mv(vec([1, 2, 3])));
print(sparse.matmul(eye(3)));
print(sparse.transpose().toDense());

final fromTriplets = SparseMatrix.fromTriplets(3, 3, [
  SparseEntry(0, 0, 1),
  SparseEntry(0, 2, 2),
  SparseEntry(2, 1, 4),
]);

final compact = mat([
  [1, 0],
  [0, 2],
]).toSparse();
```

Sparse matrices use CSR storage. The sparse API is separate from
`Matrix64` / `Matrix32` dense kernels and is intended for matrices where nonzero
entries are a small fraction of total entries.

## Precision, Numerics, And Performance Policy

Recommended choices:

- Use `Matrix64` / `Vector64` by default.
- Use `Matrix32` / `Vector32` for large throughput-oriented workloads when
  float32 error is acceptable.
- Set explicit tolerances for ill-conditioned matrices, rank decisions,
  near-singular systems, and iterative solvers.

`Matrix32` and `Vector32` use `Float32x4` on suitable hot paths, including
matrix multiplication, matrix-vector multiplication, vector-matrix
multiplication, elementwise arithmetic, scalar arithmetic, dot products, sums,
and norms. `Matrix64` / `Vector64` use `Float64List` and `Float64x2`-oriented
kernels.

API scope:

- Covers common dense-matrix workflows: construction, indexing, shape
  operations, arithmetic, statistics, JSON, decompositions, and solves.
- Adds QR, SVD, pseudoinverse, least squares, PCA, iterative methods, Krylov
  methods, and CSR sparse matrices.
- Does not include native BLAS/LAPACK, GPU execution, autodiff, distributed
  matrices, or full complex nonsymmetric eigensolvers.

## Benchmark

Benchmark scripts use an AOT runner by default. JIT is for smoke/debug runs and
must not be used for published performance claims.

Focused matrix multiplication comparison:

```bash
dart run test/benchmark_matmul.dart 256,512,1024 3 1 build/performance_report.md
```

Broader benchmark suite:

```bash
dart run test/benchmark_suite.dart 100,256,512,1000 3 1 build/benchmark_suite.md
```

The suite covers construction, scalar/elementwise operations, broadcasting,
square and rectangular matrix multiplication, matrix-vector, vector-matrix,
transpose, direct solves, vectors, sparse matrices, decompositions, least
squares, statistics, and iterative/Krylov algorithms.

Generated reports:

- [build/performance_report.md](build/performance_report.md)
- [build/benchmark_suite.md](build/benchmark_suite.md)

Performance claims must name:

- Dart SDK and command line.
- AOT/JIT mode.
- Precision: float32 or float64.
- Matrix size and shape.
- Compared package version and interface.
- Iterations, warmups, and sample count.

## Testing And Quality Gates

Routine checks:

```bash
dart format --output=none --set-exit-if-changed lib test
dart analyze
dart test
```

Optional performance regression:

```bash
MATRICES_PERF_REGRESSION=1 dart test test/performance_regression_test.dart
```

The test suite covers:

- Construction, indexing, shape validation, and error paths.
- `Matrix64` / `Matrix32` / `Vector64` / `Vector32` arithmetic.
- Matrix multiplication against reference implementations.
- LU, QR, Cholesky, determinant, inverse, and solve.
- Symmetric eigen, SVD, pseudoinverse, least squares, PCA.
- Jacobi, Gauss-Seidel, SOR, CG, GMRES, Arnoldi, power iteration.
- CSR sparse construction, conversion, transpose, serialization, vector
  multiply, and dense multiply.
- Randomized property tests, ill-conditioned Hilbert residuals, near-singular
  tolerance behavior, and float32 tolerances.

Recommended quality gates should at least include format checks, static
analysis, tests, benchmark smoke, coverage artifacts, and API reference
generation.

## Error Handling

Matrices fails fast on shape and numerical preconditions:

```dart
mat([[1, 2]]) * mat([[1, 2]]);       // ArgumentError
mat([[1, 2], [2, 4]]).inverse;       // StateError
mat([[1, 2], [3, 4]]).cholesky();    // StateError
```

The package does not silently reshape, pad, truncate, or change precision.
