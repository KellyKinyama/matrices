import 'dart:math' as math;

import 'matrix.dart';
import 'vector.dart';

class IterativeResult {
  final Vector64 solution;
  final int iterations;
  final double residualNorm;
  final bool converged;

  const IterativeResult({
    required this.solution,
    required this.iterations,
    required this.residualNorm,
    required this.converged,
  });
}

enum StationaryMethod { jacobi, gaussSeidel, sor }

extension MatrixIterativeSolvers on Matrix64 {
  IterativeResult jacobi(
    Vector64 b, {
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => _stationary(
    b,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
    method: StationaryMethod.jacobi,
  );

  IterativeResult gaussSeidel(
    Vector64 b, {
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => _stationary(
    b,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
    method: StationaryMethod.gaussSeidel,
  );

  IterativeResult sor(
    Vector64 b, {
    double omega = 1.0,
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) {
    if (omega <= 0 || omega >= 2) {
      throw ArgumentError.value(
        omega,
        'omega',
        'SOR relaxation factor must be in (0, 2).',
      );
    }
    return _stationary(
      b,
      initialGuess: initialGuess,
      maxIterations: maxIterations,
      tolerance: tolerance,
      method: StationaryMethod.sor,
      omega: omega,
    );
  }

  IterativeResult _stationary(
    Vector64 b, {
    required StationaryMethod method,
    required int maxIterations,
    required double tolerance,
    required Vector64? initialGuess,
    double omega = 1.0,
  }) {
    _checkLinearSystem(b);
    var x =
        initialGuess?.toList(growable: false) ??
        List<double>.filled(columns, 0);
    if (x.length != columns) {
      throw ArgumentError(
        'Initial guess length ${x.length} does not match $columns.',
      );
    }
    final next = List<double>.filled(columns, 0);
    var residual = double.infinity;

    for (var iteration = 1; iteration <= maxIterations; iteration++) {
      if (method == StationaryMethod.jacobi) {
        next.setAll(0, x);
      }
      for (var row = 0; row < rows; row++) {
        final diagonal = this(row, row);
        if (diagonal == 0) {
          throw StateError('Zero diagonal entry at row $row.');
        }
        var sigma = 0.0;
        for (var col = 0; col < columns; col++) {
          if (col == row) {
            continue;
          }
          final source = method == StationaryMethod.jacobi || col > row
              ? x[col]
              : next[col];
          sigma += this(row, col) * source;
        }
        final candidate = (b[row] - sigma) / diagonal;
        next[row] = method == StationaryMethod.sor
            ? (1 - omega) * x[row] + omega * candidate
            : candidate;
      }
      x = List<double>.from(next);
      residual = _residualNorm(x, b);
      if (residual <= tolerance) {
        return IterativeResult(
          solution: Vector64(x),
          iterations: iteration,
          residualNorm: residual,
          converged: true,
        );
      }
    }
    return IterativeResult(
      solution: Vector64(x),
      iterations: maxIterations,
      residualNorm: residual,
      converged: false,
    );
  }

  void _checkLinearSystem(Vector64 b) {
    if (!isSquare) {
      throw StateError('Iterative solvers require a square matrix.');
    }
    if (b.length != rows) {
      throw ArgumentError('Vector64 length ${b.length} does not match $rows.');
    }
  }

  double _residualNorm(List<double> x, Vector64 b) {
    var sum = 0.0;
    for (var row = 0; row < rows; row++) {
      var ax = 0.0;
      for (var col = 0; col < columns; col++) {
        ax += this(row, col) * x[col];
      }
      final diff = b[row] - ax;
      sum += diff * diff;
    }
    return math.sqrt(sum);
  }
}
