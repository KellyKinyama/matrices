import 'dart:math' as math;
import 'dart:typed_data';

import 'matrix.dart';

class SquareMatrix extends Matrix64 {
  final int size;

  SquareMatrix.storage(this.size, Float64List data, {bool copy = true})
    : super.storage(size, size, data, copy: copy);

  factory SquareMatrix(Iterable<Iterable<num>> rows) =>
      SquareMatrix.fromRows(rows);

  factory SquareMatrix.fromRows(Iterable<Iterable<num>> rows) {
    final matrix = Matrix64.fromRows(rows);
    return SquareMatrix.fromMatrix(matrix);
  }

  factory SquareMatrix.fromList(List<List<num>> rows) =>
      SquareMatrix.fromRows(rows);

  factory SquareMatrix.fromMatrix(Matrix64 matrix) {
    if (!matrix.isSquare) {
      throw ArgumentError(
        'Expected a square matrix, got (${matrix.rows}, ${matrix.columns}).',
      );
    }
    return SquareMatrix.storage(matrix.rows, matrix.values, copy: false);
  }

  factory SquareMatrix.identity(int size) =>
      SquareMatrix.fromMatrix(Matrix64.eye(size));

  factory SquareMatrix.eye(int size) => SquareMatrix.identity(size);

  factory SquareMatrix.zeros(int size) =>
      SquareMatrix.storage(size, Float64List(size * size), copy: false);

  factory SquareMatrix.zero(int size) => SquareMatrix.zeros(size);

  factory SquareMatrix.ones(int size) =>
      SquareMatrix.fromMatrix(Matrix64.ones(size, size));

  factory SquareMatrix.one(int size) => SquareMatrix.ones(size);

  factory SquareMatrix.full(int size, num value) =>
      SquareMatrix.fromMatrix(Matrix64.full(size, size, value));

  factory SquareMatrix.number(num value, int size) =>
      SquareMatrix.full(size, value);

  factory SquareMatrix.random(int size, {int? seed}) =>
      SquareMatrix.fromMatrix(Matrix64.random(size, size, seed: seed));

  factory SquareMatrix.diagonal(Iterable<num> diagonal) =>
      SquareMatrix.fromMatrix(Matrix64.diag(diagonal));

  factory SquareMatrix.diagonalFromNumber(num value, int size) =>
      SquareMatrix.diagonal(List<num>.filled(size, value));

  @override
  SquareMatrix get inverse => inv();

  @override
  SquareMatrix inv({double tolerance = Matrix64.defaultTolerance}) =>
      SquareMatrix.fromMatrix(super.inv(tolerance: tolerance));

  @override
  double get determinant => det();

  double get logAbsDeterminant {
    final determinant = det();
    if (determinant == 0) {
      return double.negativeInfinity;
    }
    return math.log(determinant.abs());
  }

  @override
  String toString() => 'Square${super.toString()}';
}
