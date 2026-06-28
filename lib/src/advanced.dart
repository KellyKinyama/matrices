import 'dart:math' as math;

import 'matrix.dart';
import 'matrix32.dart';
import 'vector.dart';

enum EigenMethod { powerIteration, symmetricJacobi }

class EigenPair {
  final double value;
  final Vector64 vector;

  const EigenPair({required this.value, required this.vector});

  @override
  String toString() => 'Value: $value, Vector64: $vector;';
}

class SymmetricEigenDecomposition {
  final Vector64 values;
  final Matrix64 vectors;

  const SymmetricEigenDecomposition({
    required this.values,
    required this.vectors,
  });

  Matrix64 get diagonal => Matrix64.diag(values);

  Iterable<EigenPair> get pairs sync* {
    for (var i = 0; i < values.length; i++) {
      yield EigenPair(value: values[i], vector: vectors.column(i));
    }
  }
}

class SymmetricEigenDecomposition32 {
  final Vector64 values;
  final Matrix32 vectors;

  const SymmetricEigenDecomposition32({
    required this.values,
    required this.vectors,
  });

  Matrix32 get diagonal => Matrix32.diag(values);

  Iterable<EigenPair> get pairs sync* {
    for (var i = 0; i < values.length; i++) {
      yield EigenPair(value: values[i], vector: vectors.column(i));
    }
  }
}

class SVDDecomposition {
  final Matrix64 u;
  final Vector64 singularValues;
  final Matrix64 v;

  const SVDDecomposition({
    required this.u,
    required this.singularValues,
    required this.v,
  });

  Matrix64 get s => Matrix64.diag(singularValues);
}

class SVDDecomposition32 {
  final Matrix32 u;
  final Vector64 singularValues;
  final Matrix32 v;

  const SVDDecomposition32({
    required this.u,
    required this.singularValues,
    required this.v,
  });

  Matrix32 get s => Matrix32.diag(singularValues);
}

class PCAResult {
  final Matrix64 centered;
  final Matrix64 components;
  final Vector64 explainedVariance;
  final Matrix64 transformed;

  const PCAResult({
    required this.centered,
    required this.components,
    required this.explainedVariance,
    required this.transformed,
  });
}

class PCAResult32 {
  final Matrix32 centered;
  final Matrix32 components;
  final Vector64 explainedVariance;
  final Matrix32 transformed;

  const PCAResult32({
    required this.centered,
    required this.components,
    required this.explainedVariance,
    required this.transformed,
  });
}

extension MatrixAdvancedAlgorithms on Matrix64 {
  Iterable<EigenPair> eigen({
    EigenMethod method = EigenMethod.symmetricJacobi,
    Vector64? initial,
    int iterationCount = 100,
    int? seed,
    double tolerance = 1e-12,
  }) {
    switch (method) {
      case EigenMethod.symmetricJacobi:
        return eigenSymmetric(
          maxIterations: iterationCount,
          tolerance: tolerance,
        ).pairs;
      case EigenMethod.powerIteration:
        return [
          _powerIterationEigen(
            initial: initial,
            iterationCount: iterationCount,
            seed: seed,
            tolerance: tolerance,
          ),
        ];
    }
  }

  SymmetricEigenDecomposition eigenSymmetric({
    int maxIterations = 100,
    double tolerance = 1e-12,
  }) {
    _checkSymmetric(tolerance);
    final n = rows;
    final a = toRows();
    final v = Matrix64.eye(n).toRows();

    for (var iteration = 0; iteration < maxIterations; iteration++) {
      var p = 0;
      var q = 1;
      var maxOffDiagonal = 0.0;
      for (var row = 0; row < n; row++) {
        for (var col = row + 1; col < n; col++) {
          final value = a[row][col].abs();
          if (value > maxOffDiagonal) {
            maxOffDiagonal = value;
            p = row;
            q = col;
          }
        }
      }
      if (maxOffDiagonal <= tolerance) {
        break;
      }

      final app = a[p][p];
      final aqq = a[q][q];
      final apq = a[p][q];
      final tau = (aqq - app) / (2 * apq);
      final t = tau >= 0
          ? 1 / (tau + math.sqrt(1 + tau * tau))
          : -1 / (-tau + math.sqrt(1 + tau * tau));
      final c = 1 / math.sqrt(1 + t * t);
      final s = t * c;

      for (var k = 0; k < n; k++) {
        if (k != p && k != q) {
          final akp = a[k][p];
          final akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[p][k] = a[k][p];
          a[k][q] = s * akp + c * akq;
          a[q][k] = a[k][q];
        }
      }
      a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      a[p][q] = 0;
      a[q][p] = 0;

      for (var k = 0; k < n; k++) {
        final vkp = v[k][p];
        final vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }

    final order = List<int>.generate(n, (index) => index)
      ..sort((left, right) => a[right][right].compareTo(a[left][left]));
    final values = Vector64(order.map((index) => a[index][index]));
    final vectors = Matrix64.fromColumns(
      order.map((index) => List<double>.generate(n, (row) => v[row][index])),
    );
    return SymmetricEigenDecomposition(values: values, vectors: vectors);
  }

  SVDDecomposition svd({double tolerance = 1e-12}) {
    final gram = transpose * this as Matrix64;
    final eigen = gram.eigenSymmetric(tolerance: tolerance);
    final singularValues = Vector64(
      eigen.values.map((value) => math.sqrt(math.max(value, 0))),
    );
    final k = math.min(rows, columns);
    final vThin = eigen.vectors.slice(columnStart: 0, columnEnd: k);
    final uColumns = <Vector64>[];
    for (var i = 0; i < k; i++) {
      final sigma = singularValues[i];
      if (sigma <= tolerance) {
        uColumns.add(Vector64.zeros(rows));
      } else {
        uColumns.add(mv(vThin.column(i)) / sigma);
      }
    }
    return SVDDecomposition(
      u: Matrix64.fromColumns(uColumns),
      singularValues: Vector64(singularValues.take(k)),
      v: vThin,
    );
  }

  Matrix64 pseudoInverse({double tolerance = 1e-10}) {
    final decomposition = svd(tolerance: tolerance);
    final inverseSingularValues = decomposition.singularValues.map(
      (value) => value > tolerance ? 1 / value : 0.0,
    );
    return decomposition.v *
        Matrix64.diag(inverseSingularValues) *
        decomposition.u.transpose;
  }

  Matrix64 leastSquares(Matrix64 rhs, {double tolerance = 1e-10}) =>
      pseudoInverse(tolerance: tolerance) * rhs;

  PCAResult pca({int? components, bool center = true}) {
    if (rows < 2) {
      throw StateError('PCA requires at least two observations.');
    }
    final means = meanByAxis();
    final centeredRows = List<List<double>>.generate(rows, (rowIndex) {
      return List<double>.generate(columns, (columnIndex) {
        final value = this(rowIndex, columnIndex);
        return center ? value - means[columnIndex] : value;
      });
    });
    final centered = Matrix64.fromRows(centeredRows);
    final covariance = (centered.transpose * centered as Matrix64) / (rows - 1);
    final eigen = covariance.eigenSymmetric();
    final count = components ?? columns;
    if (count <= 0 || count > columns) {
      throw ArgumentError.value(
        components,
        'components',
        'Component count must be in 1..$columns.',
      );
    }
    final selectedComponents = eigen.vectors.slice(
      columnStart: 0,
      columnEnd: count,
    );
    return PCAResult(
      centered: centered,
      components: selectedComponents,
      explainedVariance: Vector64(eigen.values.take(count)),
      transformed: centered * selectedComponents,
    );
  }

  void _checkSymmetric(double tolerance) {
    if (!isSquare) {
      throw StateError(
        'Symmetric eigen decomposition requires a square matrix.',
      );
    }
    for (var row = 0; row < rows; row++) {
      for (var col = row + 1; col < columns; col++) {
        if ((this(row, col) - this(col, row)).abs() > tolerance) {
          throw StateError('Matrix64 is not symmetric to working precision.');
        }
      }
    }
  }

  EigenPair _powerIterationEigen({
    Vector64? initial,
    required int iterationCount,
    int? seed,
    required double tolerance,
  }) {
    if (!isSquare) {
      throw StateError('Power iteration requires a square matrix.');
    }
    var vector = initial ?? _initialVector(rows, seed);
    if (vector.length != rows) {
      throw ArgumentError('Initial vector length must equal matrix size.');
    }
    vector = vector.normalize();
    var eigenvalue = 0.0;
    for (var iteration = 0; iteration < iterationCount; iteration++) {
      final product = mv(vector);
      final norm = product.norm();
      if (norm <= tolerance) {
        throw StateError('Power iteration encountered a zero vector.');
      }
      final next = product / norm;
      final candidate = next.dot(mv(next));
      if ((candidate - eigenvalue).abs() <= tolerance) {
        return EigenPair(value: candidate, vector: next);
      }
      eigenvalue = candidate;
      vector = next;
    }
    return EigenPair(value: eigenvalue, vector: vector);
  }
}

extension Matrix32AdvancedAlgorithms on Matrix32 {
  Iterable<EigenPair> eigen({
    EigenMethod method = EigenMethod.symmetricJacobi,
    Vector64? initial,
    int iterationCount = 100,
    int? seed,
    double tolerance = 1e-6,
  }) {
    switch (method) {
      case EigenMethod.symmetricJacobi:
        return eigenSymmetric(
          maxIterations: iterationCount,
          tolerance: tolerance,
        ).pairs;
      case EigenMethod.powerIteration:
        return [
          _powerIterationEigen32(
            initial: initial,
            iterationCount: iterationCount,
            seed: seed,
            tolerance: tolerance,
          ),
        ];
    }
  }

  SymmetricEigenDecomposition32 eigenSymmetric({
    int maxIterations = 100,
    double tolerance = 1e-8,
  }) {
    _checkSymmetric32(tolerance);
    final n = rows;
    final data = this.values;
    final v = Matrix32.eye(n).values;

    for (var iteration = 0; iteration < maxIterations; iteration++) {
      var p = 0;
      var q = 1;
      var maxOffDiagonal = 0.0;
      for (var row = 0; row < n; row++) {
        for (var col = row + 1; col < n; col++) {
          final value = data[row * n + col].abs();
          if (value > maxOffDiagonal) {
            maxOffDiagonal = value;
            p = row;
            q = col;
          }
        }
      }
      if (maxOffDiagonal <= tolerance) {
        break;
      }

      final app = data[p * n + p];
      final aqq = data[q * n + q];
      final apq = data[p * n + q];
      final tau = (aqq - app) / (2 * apq);
      final t = tau >= 0
          ? 1 / (tau + math.sqrt(1 + tau * tau))
          : -1 / (-tau + math.sqrt(1 + tau * tau));
      final c = 1 / math.sqrt(1 + t * t);
      final s = t * c;

      for (var k = 0; k < n; k++) {
        if (k != p && k != q) {
          final akp = data[k * n + p];
          final akq = data[k * n + q];
          data[k * n + p] = c * akp - s * akq;
          data[p * n + k] = data[k * n + p];
          data[k * n + q] = s * akp + c * akq;
          data[q * n + k] = data[k * n + q];
        }
      }
      data[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      data[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      data[p * n + q] = 0;
      data[q * n + p] = 0;

      for (var k = 0; k < n; k++) {
        final vkp = v[k * n + p];
        final vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq;
        v[k * n + q] = s * vkp + c * vkq;
      }
    }

    final order = List<int>.generate(n, (index) => index)
      ..sort(
        (left, right) =>
            data[right * n + right].compareTo(data[left * n + left]),
      );
    final eigenValues = Vector64(order.map((index) => data[index * n + index]));
    final vectors = Matrix32.fromColumns(
      order.map(
        (index) => List<double>.generate(n, (row) => v[row * n + index]),
      ),
    );
    return SymmetricEigenDecomposition32(values: eigenValues, vectors: vectors);
  }

  SVDDecomposition32 svd({double tolerance = 1e-6}) {
    final gram = transpose * this as Matrix32;
    final eigen = gram.eigenSymmetric(tolerance: tolerance);
    final singularValues = Vector64(
      eigen.values.map((value) => math.sqrt(math.max(value, 0))),
    );
    final k = math.min(rows, columns);
    final vThin = eigen.vectors.slice(columnStart: 0, columnEnd: k);
    final uColumns = <Vector64>[];
    for (var i = 0; i < k; i++) {
      final sigma = singularValues[i];
      if (sigma <= tolerance) {
        uColumns.add(Vector64.zeros(rows));
      } else {
        uColumns.add(mv(vThin.column(i)) / sigma);
      }
    }
    return SVDDecomposition32(
      u: Matrix32.fromColumns(uColumns),
      singularValues: Vector64(singularValues.take(k)),
      v: vThin,
    );
  }

  Matrix32 pseudoInverse({double tolerance = 1e-6}) {
    final decomposition = svd(tolerance: tolerance);
    final inverseSingularValues = decomposition.singularValues.map(
      (value) => value > tolerance ? 1 / value : 0.0,
    );
    return decomposition.v *
        Matrix32.diag(inverseSingularValues) *
        decomposition.u.transpose;
  }

  Matrix32 leastSquares(Matrix32 rhs, {double tolerance = 1e-6}) =>
      pseudoInverse(tolerance: tolerance) * rhs;

  PCAResult32 pca({int? components, bool center = true}) {
    if (rows < 2) {
      throw StateError('PCA requires at least two observations.');
    }
    final means = meanByAxis();
    final centeredRows = List<List<double>>.generate(rows, (rowIndex) {
      return List<double>.generate(columns, (columnIndex) {
        final value = this(rowIndex, columnIndex);
        return center ? value - means[columnIndex] : value;
      });
    });
    final centered = Matrix32.fromRows(centeredRows);
    final covariance = (centered.transpose * centered as Matrix32) / (rows - 1);
    final eigen = covariance.eigenSymmetric();
    final count = components ?? columns;
    if (count <= 0 || count > columns) {
      throw ArgumentError.value(
        components,
        'components',
        'Component count must be in 1..$columns.',
      );
    }
    final selectedComponents = eigen.vectors.slice(
      columnStart: 0,
      columnEnd: count,
    );
    return PCAResult32(
      centered: centered,
      components: selectedComponents,
      explainedVariance: Vector64(eigen.values.take(count)),
      transformed: centered * selectedComponents,
    );
  }

  void _checkSymmetric32(double tolerance) {
    if (!isSquare) {
      throw StateError(
        'Symmetric eigen decomposition requires a square matrix.',
      );
    }
    for (var row = 0; row < rows; row++) {
      for (var col = row + 1; col < columns; col++) {
        if ((this(row, col) - this(col, row)).abs() > tolerance) {
          throw StateError('Matrix64 is not symmetric to working precision.');
        }
      }
    }
  }

  EigenPair _powerIterationEigen32({
    Vector64? initial,
    required int iterationCount,
    int? seed,
    required double tolerance,
  }) {
    if (!isSquare) {
      throw StateError('Power iteration requires a square matrix.');
    }
    var vector = initial ?? _initialVector(rows, seed);
    if (vector.length != rows) {
      throw ArgumentError('Initial vector length must equal matrix size.');
    }
    vector = vector.normalize();
    var eigenvalue = 0.0;
    for (var iteration = 0; iteration < iterationCount; iteration++) {
      final product = mv(vector);
      final norm = product.norm();
      if (norm <= tolerance) {
        throw StateError('Power iteration encountered a zero vector.');
      }
      final next = product / norm;
      final candidate = next.dot(mv(next));
      if ((candidate - eigenvalue).abs() <= tolerance) {
        return EigenPair(value: candidate, vector: next);
      }
      eigenvalue = candidate;
      vector = next;
    }
    return EigenPair(value: eigenvalue, vector: vector);
  }
}

Vector64 _initialVector(int length, int? seed) {
  if (seed == null) {
    return Vector64.ones(length).normalize();
  }
  final random = math.Random(seed);
  return Vector64(
    List<double>.generate(length, (_) => random.nextDouble() + 1e-3),
  ).normalize();
}
