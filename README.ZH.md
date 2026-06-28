# Matrices

[English](https://github.com/abandoft/matrices) | [中文文档](https://github.com/abandoft/matrices/blob/master/README-ZH.md)

Matrices 是面向 Dart 和 Flutter 的矩阵运算与线性代数库。以高性能和脚本式使用体验为核心，提供接近 NumPy/MATLAB 的构造、索引、广播和运算符语义，同时保留明确的 32/64 位精度选择。

项目特点：

- 使用 Dart 自带的 `SIMD` API 加速运算，不依赖 FFI 或平台二进制运行时。
- 使用连续 typed-data 存储，避免 nested-list 结构带来的性能和语义问题。
- 提供 `Matrix64` / `Matrix32`、`Vector64` / `Vector32` 两套明确精度 API。
- 覆盖常用密集矩阵、CSR 稀疏矩阵、直接分解、高级分解、迭代法和 Krylov 方法。

## 快速开始

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

  print(a * b);      // Matrix64 矩阵乘法
  print(a * x);      // Matrix64 * Vector64
  print(a + 1);      // 标量广播
  print(a + x);      // 行向量广播
  print(a.transpose);
}
```

显式 32 位路径：

```dart
final a32 = mat32([
  [1, 2],
  [3, 4],
]);

final x32 = vec32([1, 1]);

print(a32 * x32);      // Vector32
print(a32 * a32);      // Matrix32
print(a32.toFloat64()); // Matrix64
```

## 类型系统

| 类型 | 存储 | 用途 |
|---|---|---|
| `Matrix64` | `Float64List` | 默认高精度密集矩阵 |
| `Matrix32` | `Float32List` | 内存和吞吐优先的密集矩阵 |
| `Vector64` | `Float64List` | 默认高精度向量 |
| `Vector32` | `Float32List` | 吞吐优先的 float32 向量 |
| `Matrix` | `typedef Matrix = Matrix64` | 脚本式短名 |
| `Vector` | `typedef Vector = Vector64` | 脚本式短名 |

`Matrix` 和 `Vector` 是 64 位别名，不是独立实现。需要明确表达精度的公共 API 和文档应优先写 `Matrix64` / `Vector64`。需要 float32 时使用 `Matrix32` / `Vector32`，库不会自动降低精度。

默认构造函数使用完整单词，短名作为脚本式别名。示例使用短名，便于交互式书写。

| 默认 API | 短别名 | 返回类型 |
|---|---|---|
| `matrix([...])` | `mat([...])` | `Matrix64` |
| `matrix32([...])` | `mat32([...])` | `Matrix32` |
| `vector([...])` | `vec([...])` | `Vector64` |
| `vector32([...])` | `vec32([...])` | `Vector32` |

辅助构造同样同时提供完整单词和短别名，例如 `vectorZeros` / `vecZeros`、`vectorLinspace32` / `vecLinspace32`。

## 构造 API

64 位矩阵：

```dart
final a = Matrix64([
  [1, 2],
  [3, 4],
]);

final b = Matrix([
  [1, 2],
  [3, 4],
]); // Matrix 是 Matrix64 的别名

final z = zeros(2, 3);
final o = ones(2, 3);
final f = full(2, 3, 9);
final r = rand(2, 3, seed: 42);
final i = eye(4);
final d = diag([1, 2, 3]);
final grid = arange(0, 12, columns: 4);
final samples = linspace(0, 1, 5);
```

32 位矩阵：

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

其他构造入口：

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

## 索引、视图和转换

```dart
final a = arange(0, 9, columns: 3);

print(a(1, 2));      // 元素访问
print(a[1][2]);      // 行视图访问
print(a.row(0));     // Vector64
print(a.column(1));  // Vector64

a.set(0, 0, 99);
a[1][1] = 42;

final rows = a.toRows();
final values = a.values;             // Float64List copy
final unsafe = a.unsafeValuesView;    // 内部存储视图，高级场景使用
final json = a.toJson();
final restored = Matrix64.fromJson(json);
```

行视图是固定长度视图。`Vector64` 和 `Vector32` 是固定长度线性代数对象，实现 `Iterable<double>`，但不是可变长 `List<double>`。

## 运算语义

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

print(a + b);          // 逐元素加法
print(a - 1);          // 标量广播
print(a * 2);          // 标量乘法
print(a * b);          // 矩阵乘法
print(a * x);          // 矩阵-向量乘
print(x * a);          // 向量-矩阵乘
print(a + x);          // 行向量广播
print(a.hadamard(b));  // Hadamard 逐元素乘法
print(a / 2);
```

`*` 的含义由右侧对象决定：

- `matrix * matrix` 是矩阵乘法。
- `matrix * vector` 是矩阵-向量乘。
- `vector * matrix` 是行向量-矩阵乘。
- `matrix * number` 和 `vector * number` 是标量乘。
- 逐元素矩阵乘法使用 `hadamard`。

shape 不合法时会立即抛出 `ArgumentError`，不会静默 reshape、填充或截断。

## Shape 和数据处理

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

行列变换和统计：

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

## 向量 API

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

辅助构造：

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

## 直接线性代数

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

方阵专用入口：

```dart
final s = SquareMatrix.fromList([
  [4, 7],
  [2, 6],
]);

print(s.determinant);
print(s.inverse);
print(s.logAbsDeterminant);
```

直接求解、行列式和逆矩阵使用主元策略。矩阵在工作精度下奇异时会抛出异常。病态矩阵应显式传入容差。

## 分解和高级算法

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

高层 API：

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

| API | 适用对象 | 典型用途 |
|---|---|---|
| `lu()` | 方阵 | 直接求解、行列式、带主元分解 |
| `qr()` | 高矩阵或满秩方阵 | 最小二乘、正交化 |
| `cholesky()` | 对称正定矩阵 | SPD 求解和分解 |
| `eigenSymmetric()` | 实对称矩阵 | 全特征值和特征向量 |
| `eigen()` | 对称全分解或幂迭代 | 高层特征接口 |
| `svd()` | 矩形或方阵 | 低秩分析、伪逆 |
| `pseudoInverse()` | 矩形或方阵 | Moore-Penrose 伪逆 |
| `leastSquares(rhs)` | 超定或满秩系统 | 最小二乘 |
| `pca()` | 样本 x 特征矩阵 | 主成分分析 |

`Matrix32` 暴露同名高层 API。矩阵型结果保持 32 位存储；标量累加在 Dart 语言规则下可能使用 `double` 临时值。

## 迭代法和 Krylov 方法

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

结果对象包含收敛状态、迭代次数、残差范数，以及解向量或特征对。迭代法对初值、容差、矩阵条件数和预期收敛性质敏感；生产场景应显式设置 `tolerance` 和 `maxIterations`。

## 稀疏矩阵

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

稀疏矩阵使用 CSR 存储。稀疏 API 与 `Matrix64` / `Matrix32` 密集 kernel 分离，适合非零元素远少于总元素的矩阵。

## 精度、数值和性能策略

推荐选择：

- 默认使用 `Matrix64` / `Vector64`。
- 大规模吞吐、内存带宽敏感、可接受 float32 误差时使用 `Matrix32` / `Vector32`。
- 病态矩阵、秩判断、近奇异系统和迭代法应显式设置容差。

`Matrix32` 和 `Vector32` 在适合的热路径上使用 `Float32x4`，包括矩阵乘法、矩阵-向量乘、向量-矩阵乘、逐元素运算、标量运算、dot、sum 和范数。`Matrix64` / `Vector64` 使用 `Float64List` 和 `Float64x2` 相关 kernel。

API 范围：

- 覆盖常用密集矩阵工作流：构造、索引、shape 操作、算术、统计、JSON、分解和求解。
- 额外提供 QR、SVD、伪逆、最小二乘、PCA、迭代法、Krylov 方法和 CSR 稀疏矩阵。
- 不包含 native BLAS/LAPACK、GPU、自动微分、分布式矩阵或通用复数非对称特征分解。

## Benchmark

性能测试默认使用 AOT runner。JIT 只用于冒烟和调试，不用于发布性能声明。

矩阵乘法重点对比：

```bash
dart run test/benchmark_matmul.dart 256,512,1024 3 1 build/performance_report.md
```

综合 benchmark suite：

```bash
dart run test/benchmark_suite.dart 100,256,512,1000 3 1 build/benchmark_suite.md
```

suite 覆盖构造、标量/逐元素、广播、方阵和矩形矩阵乘法、矩阵-向量、向量-矩阵、转置、直接求解、向量、稀疏矩阵、分解、最小二乘、统计和迭代/Krylov 算法。

生成报告的位置：

- [build/performance_report.md](build/performance_report.md)
- [build/benchmark_suite.md](build/benchmark_suite.md)

发布性能声明时必须说明：

- Dart SDK 和运行命令。
- AOT/JIT 模式。
- 数据精度：float32 或 float64。
- 矩阵尺寸和 shape。
- 对比库版本和接口。
- 迭代次数、warmup 和样本数。

## 测试和质量门禁

常规检查：

```bash
dart format --output=none --set-exit-if-changed lib test
dart analyze
dart test
```

可选性能回归：

```bash
MATRICES_PERF_REGRESSION=1 dart test test/performance_regression_test.dart
```

测试覆盖：

- 构造、索引、shape 校验和异常路径。
- `Matrix64` / `Matrix32` / `Vector64` / `Vector32` 基础算术。
- 矩阵乘法与参考实现对比。
- LU、QR、Cholesky、行列式、逆矩阵和求解。
- 对称特征分解、SVD、伪逆、最小二乘、PCA。
- Jacobi、Gauss-Seidel、SOR、CG、GMRES、Arnoldi、幂迭代。
- CSR 稀疏构造、转换、转置、序列化、向量乘和密集矩阵乘。
- 随机性质测试、病态 Hilbert 残差、近奇异容差行为和 float32 容差。

建议至少运行格式检查、静态分析、测试、benchmark smoke、coverage artifact 和 API reference 生成。

## 错误处理

Matrices 对 shape 和数值前置条件采用 fail-fast：

```dart
mat([[1, 2]]) * mat([[1, 2]]);       // ArgumentError
mat([[1, 2], [2, 4]]).inverse;       // StateError
mat([[1, 2], [3, 4]]).cholesky();    // StateError
```

库不会静默 reshape、隐式填充、隐式截断或自动改变精度。
