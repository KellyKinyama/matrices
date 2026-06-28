import 'dart:math' as math;

import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('medium randomized matrix multiplication matches reference', () {
    final random = math.Random(101);
    final leftRows = _rows(48, 40, random);
    final rightRows = _rows(40, 32, random);
    final expected = _referenceMatmul(leftRows, rightRows);

    final actual64 = Matrix64.fromRows(leftRows) * Matrix64.fromRows(rightRows);
    final actual32 = Matrix32.fromRows(leftRows) * Matrix32.fromRows(rightRows);

    for (var row = 0; row < actual64.rows; row++) {
      for (var col = 0; col < actual64.columns; col++) {
        expect(actual64(row, col), closeTo(expected[row][col], 1e-10));
        expect(actual32(row, col), closeTo(expected[row][col], 2e-4));
      }
    }
  });

  test('ill-conditioned Hilbert solve keeps residual bounded', () {
    final a = _hilbert(6);
    final xExpected = Matrix64.fromRows(List.generate(6, (_) => [1.0]));
    final b = a * xExpected;
    final x = a.solve(b, tolerance: 1e-15);
    final residual = (a * x - b).norm();

    expect(residual, lessThan(1e-9));
  });

  test('tolerance controls near-singular solve behavior', () {
    final nearSingular = Matrix64([
      [1, 1],
      [1, 1 + 1e-13],
    ]);
    final rhs = Matrix64([
      [2],
      [2 + 1e-13],
    ]);

    expect(() => nearSingular.solve(rhs), throwsStateError);
    final solved = nearSingular.solve(rhs, tolerance: 1e-15);
    expect((nearSingular * solved - rhs).norm(), lessThan(1e-12));
  });

  test('svd factors are orthogonal enough for rectangular matrices', () {
    final a = Matrix64([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 10],
      [2, 3, 4],
    ]);
    final svd = a.svd();

    final reconstructed = svd.u * svd.s * svd.v.transpose;
    final uIdentity = svd.u.transpose * svd.u;
    final vIdentity = svd.v.transpose * svd.v;

    expect((reconstructed - a).norm(), lessThan(1e-8));
    expect((uIdentity - Matrix64.eye(uIdentity.rows)).norm(), lessThan(1e-8));
    expect((vIdentity - Matrix64.eye(vIdentity.rows)).norm(), lessThan(1e-8));
  });

  test('Matrix32 advanced algorithms stay within float32 tolerances', () {
    final a = Matrix32([
      [4, 1],
      [1, 3],
    ]);
    final eigen = a.eigenSymmetric();
    final reconstructed =
        eigen.vectors * eigen.diagonal * eigen.vectors.transpose;

    expect((reconstructed - a).norm(), lessThan(1e-4));
    expect(
      a.eigen(method: EigenMethod.powerIteration).single.value,
      closeTo((7 + math.sqrt(5)) / 2, 1e-3),
    );
  });
}

Matrix64 _hilbert(int size) => Matrix64.fromRows(
  List.generate(
    size,
    (row) => List.generate(size, (col) => 1.0 / (row + col + 1)),
  ),
);

List<List<double>> _rows(int rows, int columns, math.Random random) =>
    List<List<double>>.generate(
      rows,
      (_) => List<double>.generate(columns, (_) => random.nextDouble() * 2 - 1),
    );

List<List<double>> _referenceMatmul(
  List<List<double>> a,
  List<List<double>> b,
) => List<List<double>>.generate(
  a.length,
  (row) => List<double>.generate(b.first.length, (col) {
    var sum = 0.0;
    for (var k = 0; k < b.length; k++) {
      sum += a[row][k] * b[k][col];
    }
    return sum;
  }),
);
