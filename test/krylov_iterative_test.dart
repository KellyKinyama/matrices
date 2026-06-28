import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test(
    'stationary iterative solvers converge on diagonally dominant systems',
    () {
      final a = mat([
        [4, 1],
        [2, 3],
      ]);
      final b = vec([1, 2]);

      final jacobi = a.jacobi(b, tolerance: 1e-9, maxIterations: 200);
      final gaussSeidel = a.gaussSeidel(b, tolerance: 1e-9, maxIterations: 200);
      final sor = a.sor(b, omega: 1.1, tolerance: 1e-9, maxIterations: 200);

      expect(jacobi.converged, isTrue);
      expect(gaussSeidel.converged, isTrue);
      expect(sor.converged, isTrue);
      _expectVectorClose(jacobi.solution, vec([0.1, 0.6]), epsilon: 1e-7);
      _expectVectorClose(gaussSeidel.solution, vec([0.1, 0.6]), epsilon: 1e-7);
      _expectVectorClose(sor.solution, vec([0.1, 0.6]), epsilon: 1e-7);
    },
  );

  test('conjugate gradient solves SPD systems', () {
    final a = mat([
      [4, 1],
      [1, 3],
    ]);
    final b = vec([1, 2]);
    final result = a.conjugateGradient(b, tolerance: 1e-12);

    expect(result.converged, isTrue);
    _expectVectorClose(result.solution, vec([1 / 11, 7 / 11]));
  });

  test('GMRES solves nonsymmetric systems', () {
    final a = mat([
      [3, 2],
      [1, 2],
    ]);
    final b = vec([5, 5]);
    final result = a.gmres(b, restart: 2, tolerance: 1e-10);

    expect(result.converged, isTrue);
    _expectVectorClose(result.solution, vec([0, 2.5]), epsilon: 1e-8);
  });

  test('Arnoldi builds an orthonormal Krylov basis', () {
    final a = mat([
      [2, 1],
      [0, 3],
    ]);
    final result = a.arnoldi(vec([0, 1]), 2);

    expect(result.basis, hasLength(2));
    expect(result.basis[0].dot(result.basis[1]).abs(), lessThan(1e-10));
  });

  test('power iteration finds the dominant eigenpair', () {
    final a = mat([
      [2, 0],
      [0, 5],
    ]);
    final result = a.powerIteration(tolerance: 1e-12);

    expect(result.converged, isTrue);
    expect(result.eigenvalue, closeTo(5, 1e-9));
    expect(result.eigenvector[1].abs(), closeTo(1, 1e-9));
  });
}

void _expectVectorClose(
  Vector64 actual,
  Vector64 expected, {
  double epsilon = 1e-9,
}) {
  expect(actual.length, expected.length);
  for (var i = 0; i < actual.length; i++) {
    expect(actual[i], closeTo(expected[i], epsilon));
  }
}
