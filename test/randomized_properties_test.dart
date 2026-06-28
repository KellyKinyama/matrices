import 'dart:math' as math;

import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('matrix multiplication matches a reference implementation', () {
    final random = math.Random(7);
    for (var trial = 0; trial < 12; trial++) {
      final rows = 1 + random.nextInt(5);
      final inner = 1 + random.nextInt(5);
      final columns = 1 + random.nextInt(5);
      final aRows = _rows(rows, inner, random);
      final bRows = _rows(inner, columns, random);
      final actual = mat(aRows) * mat(bRows);
      final expected = _referenceMatmul(aRows, bRows);

      for (var row = 0; row < rows; row++) {
        for (var col = 0; col < columns; col++) {
          expect(actual(row, col), closeTo(expected[row][col], 1e-10));
        }
      }
    }
  });

  test('inverse solves identity for well-conditioned random matrices', () {
    final random = math.Random(11);
    for (var trial = 0; trial < 6; trial++) {
      final rows = _rows(3, 3, random);
      for (var i = 0; i < 3; i++) {
        rows[i][i] += 4;
      }
      final a = mat(rows);
      final product = a * a.inverse;
      for (var row = 0; row < 3; row++) {
        for (var col = 0; col < 3; col++) {
          expect(product(row, col), closeTo(row == col ? 1 : 0, 1e-8));
        }
      }
    }
  });
}

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
