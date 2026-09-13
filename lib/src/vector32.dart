import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'kernels.dart';
import 'matrix32.dart';
import 'types.dart';
import 'vector.dart';

class Vector32 extends IterableBase<double> {
  final Float32List _data;

  Vector32(Iterable<num> values)
    : _data = Float32List.fromList(
        values.map((value) => value.toDouble()).toList(growable: false),
      );

  Vector32.storage(Float32List data, {bool copy = true})
    : _data = copy ? Float32List.fromList(data) : data;

  factory Vector32.zeros(int length) => Vector32.storage(Float32List(length));

  factory Vector32.ones(int length) =>
      Vector32.storage(Kernels.filled32(length, 1.0), copy: false);

  factory Vector32.full(int length, num value) =>
      Vector32.storage(Kernels.filled32(length, value.toDouble()), copy: false);

  factory Vector32.random(int length, {int? seed}) {
    final random = seed == null ? math.Random() : math.Random(seed);
    return Vector32.storage(
      Float32List.fromList(
        List<double>.generate(length, (_) => random.nextDouble()),
      ),
      copy: false,
    );
  }

  factory Vector32.range(num start, num stop, {num step = 1}) =>
      Vector32(Vector64.range(start, stop, step: step));

  factory Vector32.linspace(num start, num stop, int count) =>
      Vector32(Vector64.linspace(start, stop, count));

  factory Vector32.fromVector(Vector64 vector) => Vector32.storage(
    Float32List.fromList(vector.toList(growable: false)),
    copy: false,
  );

  factory Vector32.fromJson(Map<String, dynamic> json) =>
      Vector32((json['data'] as List).cast<num>());

  Float32List get values => Float32List.fromList(_data);

  Float32List get unsafeValuesView => _data;

  Vector64 toFloat64() => Vector64(_data);

  int get length => _data.length;

  @override
  Iterator<double> get iterator => _data.iterator;

  double operator [](int index) => _data[index];

  void operator []=(int index, double value) {
    _data[index] = value;
  }

  dynamic operator +(Object other) => _binary(other, ElementwiseOp.add);

  dynamic operator -(Object other) => _binary(other, ElementwiseOp.subtract);

  dynamic operator *(Object other) {
    if (other is Matrix32) {
      return multiplyMatrix(other);
    }
    return _binary(other, ElementwiseOp.multiply);
  }

  dynamic operator /(Object other) => _binary(other, ElementwiseOp.divide);

  /// Scalar multiply that preserves the static [Vector32] type.
  ///
  /// `operator *` returns `dynamic` because it also dispatches to the
  /// vector/matrix product; use [scale] when the result must stay
  /// statically typed for chaining.
  Vector32 scale(num factor) =>
      Vector32.storage(Kernels.scale32(_data, factor.toDouble()), copy: false);

  double dot(Vector32 other) {
    _checkLength(other);
    return Kernels.dot32(_data, 0, other._data, 0, length);
  }

  Vector32 multiplyMatrix(Matrix32 matrix) {
    if (length != matrix.rows) {
      throw ArgumentError(
        'Vector64/matrix shape mismatch: $length x '
        '(${matrix.rows}, ${matrix.columns}).',
      );
    }
    final matrixData = matrix.unsafeTransposedValuesView;
    final result = Kernels.mv32(matrixData, matrix.columns, matrix.rows, _data);
    return Vector32.storage(result, copy: false);
  }

  double get sum => Kernels.sum32(_data);

  double get mean {
    if (isEmpty) {
      throw StateError('Cannot compute the mean of an empty vector.');
    }
    return sum / length;
  }

  double norm({double p = 2}) {
    if (p <= 0) {
      throw ArgumentError.value(p, 'p', 'Norm order must be positive.');
    }
    if (p == 2) {
      return math.sqrt(dot(this));
    }
    var total = 0.0;
    for (final value in _data) {
      total += math.pow(value.abs(), p).toDouble();
    }
    return math.pow(total, 1 / p).toDouble();
  }

  double normType([VectorNorm norm = VectorNorm.euclidean]) => switch (norm) {
    VectorNorm.euclidean => this.norm(),
    VectorNorm.manhattan => fold<double>(
      0.0,
      (sum, value) => sum + value.abs(),
    ),
  };

  double distanceTo(Vector32 other, [Distance distance = Distance.euclidean]) {
    _checkLength(other);
    final difference = this - other as Vector32;
    return switch (distance) {
      Distance.euclidean => difference.norm(),
      Distance.manhattan => difference.fold<double>(
        0.0,
        (sum, value) => sum + value.abs(),
      ),
      Distance.cosine => 1 - cosine(other),
      Distance.hamming => _hamming(other),
    };
  }

  double cosine(Vector32 other) {
    _checkLength(other);
    final denominator = norm() * other.norm();
    if (denominator == 0) {
      throw StateError('Cannot compute cosine with a zero vector.');
    }
    return dot(other) / denominator;
  }

  Vector32 normalize([VectorNorm norm = VectorNorm.euclidean]) {
    final value = normType(norm);
    if (value == 0) {
      throw StateError('Cannot normalize a zero vector.');
    }
    return this / value;
  }

  Vector32 mapValues(double Function(double value) transform) =>
      Vector32(_data.map(transform));

  Vector32 copy() => Vector32.storage(_data);

  Map<String, dynamic> toJson() => <String, dynamic>{
    'dtype': 'float32',
    'data': toList(growable: false),
  };

  dynamic _binary(Object other, ElementwiseOp op) {
    if (other is num) {
      return Vector32.storage(
        Kernels.scalar32(_data, other.toDouble(), op),
        copy: false,
      );
    }
    if (other is Vector32) {
      _checkLength(other);
      return Vector32.storage(
        op == ElementwiseOp.add
            ? Kernels.add32(_data, other._data)
            : Kernels.elementwise32(_data, other._data, op),
        copy: false,
      );
    }
    if (other is Matrix32) {
      return _broadcastMatrix(other, op);
    }
    throw ArgumentError.value(
      other,
      'other',
      'Expected a num, Vector32, or Matrix32.',
    );
  }

  Matrix32 _broadcastMatrix(Matrix32 matrix, ElementwiseOp op) {
    if (length != matrix.columns && length != matrix.rows) {
      throw ArgumentError(
        'Vector64 length $length must match matrix rows ${matrix.rows} or '
        'columns ${matrix.columns} for broadcasting.',
      );
    }
    final rowWise = length == matrix.columns;
    final rows = List<List<double>>.generate(matrix.rows, (row) {
      return List<double>.generate(matrix.columns, (column) {
        final value = rowWise ? _data[column] : _data[row];
        final matrixValue = matrix(row, column);
        return switch (op) {
          ElementwiseOp.add => value + matrixValue,
          ElementwiseOp.subtract => value - matrixValue,
          ElementwiseOp.multiply => value * matrixValue,
          ElementwiseOp.divide => value / matrixValue,
        };
      });
    });
    return Matrix32.fromRows(rows);
  }

  void _checkLength(Vector32 other) {
    if (length != other.length) {
      throw ArgumentError(
        'Vector64 length mismatch: $length vs ${other.length}.',
      );
    }
  }

  double _hamming(Vector32 other) {
    var result = 0.0;
    for (var i = 0; i < length; i++) {
      if (this[i] != other[i]) {
        result++;
      }
    }
    return result;
  }

  @override
  String toString() => toList(growable: false).toString();
}

Vector32 vector32(Iterable<num> values) => Vector32(values);

Vector32 vec32(Iterable<num> values) => vector32(values);

Vector32 vectorZeros32(int length) => Vector32.zeros(length);

Vector32 vecZeros32(int length) => vectorZeros32(length);

Vector32 vectorOnes32(int length) => Vector32.ones(length);

Vector32 vecOnes32(int length) => vectorOnes32(length);

Vector32 vectorFull32(int length, num value) => Vector32.full(length, value);

Vector32 vecFull32(int length, num value) => vectorFull32(length, value);

Vector32 randomVector32(int length, {int? seed}) =>
    Vector32.random(length, seed: seed);

Vector32 randVec32(int length, {int? seed}) =>
    randomVector32(length, seed: seed);

Vector32 vectorRange32(num start, num stop, {num step = 1}) =>
    Vector32.range(start, stop, step: step);

Vector32 vecRange32(num start, num stop, {num step = 1}) =>
    vectorRange32(start, stop, step: step);

Vector32 vectorLinspace32(num start, num stop, int count) =>
    Vector32.linspace(start, stop, count);

Vector32 vecLinspace32(num start, num stop, int count) =>
    vectorLinspace32(start, stop, count);
