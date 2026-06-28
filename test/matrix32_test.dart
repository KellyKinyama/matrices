import 'dart:typed_data';

import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('Matrix32 stores data in Float32List', () {
    final a = mat32([
      [1.25, 2.5],
      [3.75, 4.125],
    ]);

    expect(a.values, isA<Float32List>());
    expect(a.rows, 2);
    expect(a.columns, 2);
    expect(a(1, 0), closeTo(3.75, 1e-6));
  });

  test('Matrix32 multiplies and matches float64 within f32 tolerance', () {
    final leftRows = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    final rightRows = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    final f32 = mat32(leftRows) * mat32(rightRows);
    final f64 = mat(leftRows) * mat(rightRows);

    for (var row = 0; row < f32.rows; row++) {
      for (var col = 0; col < f32.columns; col++) {
        expect(f32(row, col), closeTo(f64(row, col), 1e-4));
      }
    }
  });

  test('Matrix32 supports scalar and elementwise operations', () {
    final a = mat32([
      [1, 2],
      [3, 4],
    ]);
    final b = mat32([
      [5, 6],
      [7, 8],
    ]);

    expect((a + b)(1, 1), closeTo(12, 1e-6));
    expect((b - a)(0, 0), closeTo(4, 1e-6));
    expect((a * 2)(1, 0), closeTo(6, 1e-6));
    expect(a.hadamard(b)(0, 1), closeTo(12, 1e-6));
  });

  test(
    'Matrix32 supports Vector32 broadcasting and linear algebra operators',
    () {
      final a = mat32([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      final row = vec32([10, 20, 30, 40]);
      final column = vec32([2, 3]);
      final x = vec32([1, 1, 1, 1]);
      final left = vec32([1, 2]);

      expect(a + row, isA<Matrix32>());
      expect((a + row).toRows(), [
        [11, 22, 33, 44],
        [15, 26, 37, 48],
      ]);
      expect((row + a).toRows(), [
        [11, 22, 33, 44],
        [15, 26, 37, 48],
      ]);
      expect(a.hadamard(column).toRows(), [
        [2, 4, 6, 8],
        [15, 18, 21, 24],
      ]);
      expect((column - a).toRows(), [
        [1, 0, -1, -2],
        [-2, -3, -4, -5],
      ]);

      final y = a * x;
      final z = left * a;
      expect(y, isA<Vector32>());
      expect(y, [10, 26]);
      expect(z, isA<Vector32>());
      expect(z, [11, 14, 17, 20]);
    },
  );

  test('Vector32 uses float32 storage and SIMD-backed arithmetic', () {
    final x = vec32([1, 2, 3, 4]);
    final y = vec32([5, 6, 7, 8]);

    expect(x.values, isA<Float32List>());
    expect(x + y, [6, 8, 10, 12]);
    expect(x * 2, [2, 4, 6, 8]);
    expect(x.dot(y), closeTo(70, 1e-5));
    expect(x.normalize().norm(), closeTo(1, 1e-5));
  });

  test('Matrix32 converts explicitly to float64 Matrix64', () {
    final converted = mat32([
      [1, 2],
      [3, 4],
    ]).toFloat64();

    expect(converted, isA<Matrix64>());
    expect(converted(1, 1), 4);
  });
}
