import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('symmetric eigen decomposition reconstructs the matrix', () {
    final a = Matrix64([
      [2, 1],
      [1, 2],
    ]);

    final eigen = a.eigenSymmetric();
    final reconstructed =
        eigen.vectors * eigen.diagonal * eigen.vectors.transpose;

    expect(eigen.values[0], closeTo(3, 1e-10));
    expect(eigen.values[1], closeTo(1, 1e-10));
    expect(reconstructed(0, 0), closeTo(2, 1e-10));
    expect(reconstructed(0, 1), closeTo(1, 1e-10));
    expect(reconstructed(1, 0), closeTo(1, 1e-10));
    expect(reconstructed(1, 1), closeTo(2, 1e-10));
  });

  test('high-level eigen API supports full symmetric and dominant modes', () {
    final a = Matrix64([
      [2, 1],
      [1, 2],
    ]);

    final symmetric = a.eigen().toList();
    final dominant = a
        .eigen(method: EigenMethod.powerIteration, iterationCount: 100, seed: 7)
        .single;

    expect(symmetric, hasLength(2));
    expect(symmetric.first.value, closeTo(3, 1e-10));
    expect(symmetric.last.value, closeTo(1, 1e-10));
    expect(dominant.value, closeTo(3, 1e-8));
  });

  test('svd reconstructs a rectangular matrix', () {
    final a = Matrix64([
      [3, 0],
      [0, 2],
      [0, 0],
    ]);

    final svd = a.svd();
    final reconstructed = svd.u * svd.s * svd.v.transpose;

    expect(svd.singularValues[0], closeTo(3, 1e-10));
    expect(svd.singularValues[1], closeTo(2, 1e-10));
    expect(reconstructed.toRows(), [
      [closeTo(3, 1e-10), closeTo(0, 1e-10)],
      [closeTo(0, 1e-10), closeTo(2, 1e-10)],
      [closeTo(0, 1e-10), closeTo(0, 1e-10)],
    ]);
  });

  test('pseudo inverse satisfies Moore-Penrose reconstruction identity', () {
    final a = Matrix64([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);

    final pinv = a.pseudoInverse();
    final reconstructed = a * pinv * a;

    for (var row = 0; row < a.rows; row++) {
      for (var column = 0; column < a.columns; column++) {
        expect(reconstructed(row, column), closeTo(a(row, column), 1e-8));
      }
    }
  });

  test('least squares solves overdetermined systems', () {
    final a = Matrix64([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    final b = Matrix64([
      [1],
      [2],
      [2],
    ]);

    final x = a.leastSquares(b);

    expect(x(0, 0), closeTo(2 / 3, 1e-8));
    expect(x(1, 0), closeTo(0.5, 1e-8));
  });

  test('pca returns projected observations and explained variance', () {
    final data = Matrix64([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);

    final pca = data.pca(components: 1);

    expect(pca.components.shape, (2, 1));
    expect(pca.transformed.shape, (4, 1));
    expect(pca.explainedVariance[0], greaterThan(0));
    expect(pca.centered.meanByAxis(Axis.columns), [
      closeTo(0, 1e-12),
      closeTo(0, 1e-12),
    ]);
  });

  test('Matrix32 exposes native float32 advanced algorithm results', () {
    final a = Matrix32([
      [1, 0],
      [0, 2],
      [0, 0],
    ]);
    final b = Matrix32([
      [1],
      [4],
      [0],
    ]);

    final gram = a.transpose * a as Matrix32;
    final eigen = gram.eigenSymmetric();
    final svd = a.svd();
    final x = a.leastSquares(b);
    final pinv = a.pseudoInverse();
    final pca = Matrix32([
      [1, 2],
      [2, 3],
      [3, 4],
    ]).pca(components: 1);

    expect(eigen.vectors, isA<Matrix32>());
    expect(svd.u, isA<Matrix32>());
    expect(svd.v, isA<Matrix32>());
    expect(svd.singularValues[0], closeTo(2, 1e-5));
    expect(x(0, 0), closeTo(1, 1e-5));
    expect(x(1, 0), closeTo(2, 1e-5));
    expect(pinv.shape, (2, 3));
    expect(pca.components, isA<Matrix32>());
    expect(pca.transformed.shape, (3, 1));
  });
}
