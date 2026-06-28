import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('LU reconstructs a pivoted matrix and solves systems', () {
    final a = mat([
      [0, 2, 1],
      [1, 1, 0],
      [2, 0, 1],
    ]);
    final lu = a.lu();

    _expectMatrixClose(lu.lower * lu.upper, lu.permutation * a);

    final b = mat([
      [1],
      [2],
      [3],
    ]);
    final x = lu.solve(b);
    _expectMatrixClose(a * x, b);
  });

  test('QR reconstructs a tall matrix and solves least squares', () {
    final a = mat([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    final qr = a.qr();

    _expectMatrixClose(qr.q * qr.r, a);
    _expectMatrixClose(qr.q.transpose * qr.q, eye(2));

    final b = mat([
      [1],
      [2],
      [2],
    ]);
    final x = qr.solve(b);
    _expectClose(x(0, 0), 2 / 3);
    _expectClose(x(1, 0), 0.5);
  });

  test('Cholesky reconstructs SPD matrices and solves systems', () {
    final a = mat([
      [4, 2],
      [2, 3],
    ]);
    final decomposition = a.cholesky();

    _expectMatrixClose(decomposition.lower * decomposition.lower.transpose, a);

    final b = mat([
      [1],
      [1],
    ]);
    _expectMatrixClose(a * decomposition.solve(b), b);
  });

  test('Direct determinant and inverse agree with decompositions', () {
    final a = SquareMatrix.fromList([
      [4, 7],
      [2, 6],
    ]);

    _expectClose(a.determinant, 10);
    _expectMatrixClose(a * a.inverse, eye(2));
    _expectClose(a.lu().determinant(), 10);
  });
}

void _expectMatrixClose(
  Matrix64 actual,
  Matrix64 expected, {
  double epsilon = 1e-9,
}) {
  expect(actual.rows, expected.rows);
  expect(actual.columns, expected.columns);
  for (var row = 0; row < actual.rows; row++) {
    for (var col = 0; col < actual.columns; col++) {
      _expectClose(actual(row, col), expected(row, col), epsilon: epsilon);
    }
  }
}

void _expectClose(double actual, double expected, {double epsilon = 1e-9}) {
  expect(actual, closeTo(expected, epsilon));
}
