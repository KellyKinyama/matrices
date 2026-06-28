import 'dart:math' as math;

import 'decompositions.dart';
import 'matrix.dart';
import 'vector.dart';

class ArnoldiResult {
  final List<Vector64> basis;
  final Matrix64 hessenberg;

  const ArnoldiResult({required this.basis, required this.hessenberg});
}

class KrylovResult {
  final Vector64 solution;
  final int iterations;
  final double residualNorm;
  final bool converged;

  const KrylovResult({
    required this.solution,
    required this.iterations,
    required this.residualNorm,
    required this.converged,
  });
}

class PowerIterationResult {
  final double eigenvalue;
  final Vector64 eigenvector;
  final int iterations;
  final bool converged;

  const PowerIterationResult({
    required this.eigenvalue,
    required this.eigenvector,
    required this.iterations,
    required this.converged,
  });
}

extension MatrixKrylovMethods on Matrix64 {
  ArnoldiResult arnoldi(
    Vector64 start,
    int dimension, {
    double tolerance = Matrix64.defaultTolerance,
  }) {
    if (!isSquare) {
      throw StateError('Arnoldi iteration requires a square matrix.');
    }
    if (start.length != rows) {
      throw ArgumentError('Start vector length ${start.length} must be $rows.');
    }
    if (dimension <= 0) {
      throw ArgumentError.value(
        dimension,
        'dimension',
        'Dimension must be positive.',
      );
    }

    final q = <Vector64>[];
    final h = List<List<double>>.generate(
      dimension + 1,
      (_) => List<double>.filled(dimension, 0),
    );
    final startNorm = start.norm();
    if (startNorm <= tolerance) {
      throw ArgumentError.value(
        start,
        'start',
        'Start vector must be nonzero.',
      );
    }
    q.add(start / startNorm);

    for (var k = 0; k < dimension; k++) {
      var v = mv(q[k]);
      for (var j = 0; j <= k; j++) {
        h[j][k] = q[j].dot(v);
        v = v - q[j] * h[j][k];
      }
      h[k + 1][k] = v.norm();
      if (h[k + 1][k] <= tolerance) {
        return ArnoldiResult(
          basis: q,
          hessenberg: Matrix64.fromRows(
            h.take(k + 2).map((row) => row.take(k + 1)),
          ),
        );
      }
      if (k + 1 < dimension) {
        q.add(v / h[k + 1][k]);
      }
    }
    return ArnoldiResult(basis: q, hessenberg: Matrix64.fromRows(h));
  }

  KrylovResult conjugateGradient(
    Vector64 b, {
    Vector64? initialGuess,
    int? maxIterations,
    double tolerance = 1e-10,
  }) {
    if (!isSquare) {
      throw StateError('Conjugate gradient requires a square matrix.');
    }
    if (b.length != rows) {
      throw ArgumentError('Vector64 length ${b.length} does not match $rows.');
    }
    final limit = maxIterations ?? rows * 2;
    var x = initialGuess ?? Vector64.zeros(columns);
    if (x.length != columns) {
      throw ArgumentError(
        'Initial guess length ${x.length} does not match $columns.',
      );
    }
    var r = b - mv(x);
    var p = r.copy();
    var rsOld = r.dot(r);
    if (math.sqrt(rsOld) <= tolerance) {
      return KrylovResult(
        solution: x,
        iterations: 0,
        residualNorm: math.sqrt(rsOld),
        converged: true,
      );
    }

    for (var iteration = 1; iteration <= limit; iteration++) {
      final ap = mv(p);
      final denominator = p.dot(ap);
      if (denominator.abs() <= Matrix64.defaultTolerance) {
        break;
      }
      final alpha = rsOld / denominator;
      x = x + p * alpha;
      r = r - ap * alpha;
      final rsNew = r.dot(r);
      final residual = math.sqrt(rsNew);
      if (residual <= tolerance) {
        return KrylovResult(
          solution: x,
          iterations: iteration,
          residualNorm: residual,
          converged: true,
        );
      }
      p = r + p * (rsNew / rsOld);
      rsOld = rsNew;
    }

    final residual = (b - mv(x)).norm();
    return KrylovResult(
      solution: x,
      iterations: limit,
      residualNorm: residual,
      converged: false,
    );
  }

  KrylovResult gmres(
    Vector64 b, {
    Vector64? initialGuess,
    int restart = 30,
    int maxIterations = 300,
    double tolerance = 1e-10,
  }) {
    if (!isSquare) {
      throw StateError('GMRES requires a square matrix.');
    }
    if (b.length != rows) {
      throw ArgumentError('Vector64 length ${b.length} does not match $rows.');
    }
    if (restart <= 0) {
      throw ArgumentError.value(
        restart,
        'restart',
        'Restart must be positive.',
      );
    }

    var x = initialGuess ?? Vector64.zeros(columns);
    var totalIterations = 0;
    var residual = (b - mv(x)).norm();
    if (residual <= tolerance) {
      return KrylovResult(
        solution: x,
        iterations: 0,
        residualNorm: residual,
        converged: true,
      );
    }

    while (totalIterations < maxIterations) {
      final r = b - mv(x);
      final beta = r.norm();
      if (beta <= tolerance) {
        return KrylovResult(
          solution: x,
          iterations: totalIterations,
          residualNorm: beta,
          converged: true,
        );
      }

      final q = <Vector64>[r / beta];
      final h = List<List<double>>.generate(
        restart + 1,
        (_) => List<double>.filled(restart, 0),
      );
      var used = 0;
      for (
        var k = 0;
        k < restart && totalIterations < maxIterations;
        k++, totalIterations++
      ) {
        var v = mv(q[k]);
        for (var j = 0; j <= k; j++) {
          h[j][k] = q[j].dot(v);
          v = v - q[j] * h[j][k];
        }
        h[k + 1][k] = v.norm();
        used = k + 1;
        if (h[k + 1][k] > Matrix64.defaultTolerance && k + 1 < restart) {
          q.add(v / h[k + 1][k]);
        }

        final y = _leastSquaresForGmres(h, beta, used);
        var candidate = x.copy();
        for (var i = 0; i < used; i++) {
          candidate = candidate + q[i] * y[i];
        }
        residual = (b - mv(candidate)).norm();
        if (residual <= tolerance) {
          return KrylovResult(
            solution: candidate,
            iterations: totalIterations + 1,
            residualNorm: residual,
            converged: true,
          );
        }
      }

      final y = _leastSquaresForGmres(h, beta, used);
      for (var i = 0; i < used; i++) {
        x = x + q[i] * y[i];
      }
      residual = (b - mv(x)).norm();
    }

    return KrylovResult(
      solution: x,
      iterations: totalIterations,
      residualNorm: residual,
      converged: false,
    );
  }

  PowerIterationResult powerIteration({
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) {
    if (!isSquare) {
      throw StateError('Power iteration requires a square matrix.');
    }
    var vector = initialGuess ?? Vector64.ones(rows);
    if (vector.length != rows) {
      throw ArgumentError(
        'Initial vector length ${vector.length} does not match $rows.',
      );
    }
    vector = vector / vector.norm();
    var eigenvalue = 0.0;
    for (var iteration = 1; iteration <= maxIterations; iteration++) {
      final next = mv(vector);
      final norm = next.norm();
      if (norm <= Matrix64.defaultTolerance) {
        throw StateError('Power iteration encountered a zero vector.');
      }
      final normalized = next / norm;
      final candidate = normalized.dot(mv(normalized));
      if ((candidate - eigenvalue).abs() <= tolerance) {
        return PowerIterationResult(
          eigenvalue: candidate,
          eigenvector: normalized,
          iterations: iteration,
          converged: true,
        );
      }
      eigenvalue = candidate;
      vector = normalized;
    }
    return PowerIterationResult(
      eigenvalue: eigenvalue,
      eigenvector: vector,
      iterations: maxIterations,
      converged: false,
    );
  }

  List<double> _leastSquaresForGmres(
    List<List<double>> h,
    double beta,
    int used,
  ) {
    final rows = used + 1;
    final smallH = Matrix64.fromRows(
      List<List<double>>.generate(
        rows,
        (row) => List<double>.generate(used, (col) => h[row][col]),
      ),
    );
    final rhs = Matrix64.fromRows(
      List<List<double>>.generate(rows, (row) => [row == 0 ? beta : 0.0]),
    );
    return smallH.qr().solve(rhs).column(0).toList(growable: false);
  }
}
