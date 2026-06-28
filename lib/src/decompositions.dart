import 'dart:math' as math;

import 'matrix.dart';

class LUDecomposition {
  final Matrix64 lower;
  final Matrix64 upper;
  final List<int> pivots;
  final int pivotSign;

  const LUDecomposition({
    required this.lower,
    required this.upper,
    required this.pivots,
    required this.pivotSign,
  });

  Matrix64 get permutation {
    final rows = List<List<double>>.generate(
      pivots.length,
      (row) => List<double>.filled(pivots.length, 0),
    );
    for (var row = 0; row < pivots.length; row++) {
      rows[row][pivots[row]] = 1;
    }
    return Matrix64.fromRows(rows);
  }

  double determinant() {
    var result = pivotSign.toDouble();
    for (var i = 0; i < upper.rows; i++) {
      result *= upper(i, i);
    }
    return result;
  }

  Matrix64 solve(Matrix64 rhs) {
    if (rhs.rows != pivots.length) {
      throw ArgumentError(
        'Right-hand side row count ${rhs.rows} does not match ${pivots.length}.',
      );
    }
    final n = pivots.length;
    final columns = rhs.columns;
    final y = List<List<double>>.generate(
      n,
      (row) => List<double>.generate(columns, (col) => rhs(pivots[row], col)),
    );

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < i; j++) {
        final factor = lower(i, j);
        if (factor == 0) {
          continue;
        }
        for (var col = 0; col < columns; col++) {
          y[i][col] -= factor * y[j][col];
        }
      }
    }

    for (var i = n - 1; i >= 0; i--) {
      final diagonal = upper(i, i);
      if (diagonal == 0) {
        throw StateError('Matrix64 is singular.');
      }
      for (var col = 0; col < columns; col++) {
        for (var j = i + 1; j < n; j++) {
          y[i][col] -= upper(i, j) * y[j][col];
        }
        y[i][col] /= diagonal;
      }
    }

    return Matrix64.fromRows(y);
  }
}

class QRDecomposition {
  final Matrix64 q;
  final Matrix64 r;

  const QRDecomposition({required this.q, required this.r});

  Matrix64 solve(Matrix64 rhs, {double tolerance = Matrix64.defaultTolerance}) {
    if (rhs.rows != q.rows) {
      throw ArgumentError(
        'Right-hand side row count ${rhs.rows} does not match ${q.rows}.',
      );
    }
    final qtB = q.transpose * rhs;
    final n = r.rows;
    final result = List<List<double>>.generate(
      n,
      (_) => List<double>.filled(rhs.columns, 0),
    );
    for (var col = 0; col < rhs.columns; col++) {
      for (var i = n - 1; i >= 0; i--) {
        var value = qtB(i, col);
        for (var j = i + 1; j < n; j++) {
          value -= r(i, j) * result[j][col];
        }
        final diagonal = r(i, i);
        if (diagonal.abs() <= tolerance) {
          throw StateError('Matrix64 is rank deficient to working precision.');
        }
        result[i][col] = value / diagonal;
      }
    }
    return Matrix64.fromRows(result);
  }
}

class CholeskyDecomposition {
  final Matrix64 lower;

  const CholeskyDecomposition(this.lower);

  Matrix64 get upper => lower.transpose;

  double determinant() {
    var product = 1.0;
    for (var i = 0; i < lower.rows; i++) {
      product *= lower(i, i);
    }
    return product * product;
  }

  Matrix64 solve(Matrix64 rhs, {double tolerance = Matrix64.defaultTolerance}) {
    if (rhs.rows != lower.rows) {
      throw ArgumentError(
        'Right-hand side row count ${rhs.rows} does not match ${lower.rows}.',
      );
    }
    final n = lower.rows;
    final columns = rhs.columns;
    final y = List<List<double>>.generate(
      n,
      (row) => List<double>.generate(columns, (col) => rhs(row, col)),
    );

    for (var i = 0; i < n; i++) {
      final diagonal = lower(i, i);
      if (diagonal.abs() <= tolerance) {
        throw StateError('Matrix64 is not positive definite.');
      }
      for (var col = 0; col < columns; col++) {
        for (var j = 0; j < i; j++) {
          y[i][col] -= lower(i, j) * y[j][col];
        }
        y[i][col] /= diagonal;
      }
    }

    for (var i = n - 1; i >= 0; i--) {
      final diagonal = lower(i, i);
      for (var col = 0; col < columns; col++) {
        for (var j = i + 1; j < n; j++) {
          y[i][col] -= lower(j, i) * y[j][col];
        }
        y[i][col] /= diagonal;
      }
    }

    return Matrix64.fromRows(y);
  }
}

extension MatrixDecompositions on Matrix64 {
  LUDecomposition lu({double tolerance = Matrix64.defaultTolerance}) {
    if (!isSquare) {
      throw StateError('LU decomposition requires a square matrix.');
    }
    final n = rows;
    final a = toRows();
    final pivots = List<int>.generate(n, (index) => index);
    var pivotSign = 1;

    for (var k = 0; k < n; k++) {
      var pivotRow = k;
      var maxValue = a[k][k].abs();
      for (var row = k + 1; row < n; row++) {
        final value = a[row][k].abs();
        if (value > maxValue) {
          maxValue = value;
          pivotRow = row;
        }
      }
      if (maxValue <= tolerance) {
        throw StateError('Matrix64 is singular to working precision.');
      }
      if (pivotRow != k) {
        final tmp = a[k];
        a[k] = a[pivotRow];
        a[pivotRow] = tmp;
        final pivotTmp = pivots[k];
        pivots[k] = pivots[pivotRow];
        pivots[pivotRow] = pivotTmp;
        pivotSign = -pivotSign;
      }
      for (var row = k + 1; row < n; row++) {
        a[row][k] /= a[k][k];
        final factor = a[row][k];
        for (var col = k + 1; col < n; col++) {
          a[row][col] -= factor * a[k][col];
        }
      }
    }

    final lower = List<List<double>>.generate(
      n,
      (_) => List<double>.filled(n, 0),
    );
    final upper = List<List<double>>.generate(
      n,
      (_) => List<double>.filled(n, 0),
    );
    for (var row = 0; row < n; row++) {
      for (var col = 0; col < n; col++) {
        if (row > col) {
          lower[row][col] = a[row][col];
        } else {
          upper[row][col] = a[row][col];
          if (row == col) {
            lower[row][col] = 1;
          }
        }
      }
    }
    return LUDecomposition(
      lower: Matrix64.fromRows(lower),
      upper: Matrix64.fromRows(upper),
      pivots: pivots,
      pivotSign: pivotSign,
    );
  }

  QRDecomposition qr({double tolerance = Matrix64.defaultTolerance}) {
    if (rows < columns) {
      throw StateError('Thin QR decomposition requires rows >= columns.');
    }
    final qColumns = <List<double>>[];
    final r = List<List<double>>.generate(
      columns,
      (_) => List<double>.filled(columns, 0),
    );

    for (var col = 0; col < columns; col++) {
      final v = List<double>.generate(rows, (row) => this(row, col));
      for (var previous = 0; previous < qColumns.length; previous++) {
        var projection = 0.0;
        final q = qColumns[previous];
        for (var row = 0; row < rows; row++) {
          projection += q[row] * v[row];
        }
        r[previous][col] = projection;
        for (var row = 0; row < rows; row++) {
          v[row] -= projection * q[row];
        }
      }
      var norm = 0.0;
      for (final value in v) {
        norm += value * value;
      }
      norm = math.sqrt(norm);
      if (norm <= tolerance) {
        throw StateError('Matrix64 is rank deficient to working precision.');
      }
      r[col][col] = norm;
      for (var row = 0; row < rows; row++) {
        v[row] /= norm;
      }
      qColumns.add(v);
    }

    final qRows = List<List<double>>.generate(
      rows,
      (row) => List<double>.generate(columns, (col) => qColumns[col][row]),
    );
    return QRDecomposition(
      q: Matrix64.fromRows(qRows),
      r: Matrix64.fromRows(r),
    );
  }

  CholeskyDecomposition cholesky({
    double tolerance = Matrix64.defaultTolerance,
  }) {
    if (!isSquare) {
      throw StateError('Cholesky decomposition requires a square matrix.');
    }
    final n = rows;
    final lower = List<List<double>>.generate(
      n,
      (_) => List<double>.filled(n, 0),
    );
    for (var row = 0; row < n; row++) {
      for (var col = 0; col <= row; col++) {
        var sum = this(row, col);
        for (var k = 0; k < col; k++) {
          sum -= lower[row][k] * lower[col][k];
        }
        if (row == col) {
          if (sum <= tolerance) {
            throw StateError('Matrix64 is not positive definite.');
          }
          lower[row][col] = math.sqrt(sum);
        } else {
          lower[row][col] = sum / lower[col][col];
        }
      }
    }
    return CholeskyDecomposition(Matrix64.fromRows(lower));
  }
}
