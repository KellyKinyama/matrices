import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'kernels.dart';
import 'matrix.dart';
import 'types.dart';

class Vector64 extends IterableBase<double> {
  final Float64List _data;

  Vector64(Iterable<num> values)
    : _data = Float64List.fromList(
        values.map((value) => value.toDouble()).toList(growable: false),
      );

  Vector64.storage(Float64List data, {bool copy = true})
    : _data = copy ? Float64List.fromList(data) : data;

  factory Vector64.zeros(int length) => Vector64.storage(Float64List(length));

  factory Vector64.ones(int length) =>
      Vector64.storage(Kernels.filled(length, 1.0), copy: false);

  factory Vector64.full(int length, num value) =>
      Vector64.storage(Kernels.filled(length, value.toDouble()), copy: false);

  factory Vector64.random(int length, {int? seed}) {
    final random = seed == null ? math.Random() : math.Random(seed);
    return Vector64.storage(
      Float64List.fromList(
        List<double>.generate(length, (_) => random.nextDouble()),
      ),
      copy: false,
    );
  }

  factory Vector64.range(num start, num stop, {num step = 1}) {
    final delta = step.toDouble();
    if (delta == 0) {
      throw ArgumentError.value(step, 'step', 'Step must not be zero.');
    }
    final result = <double>[];
    var current = start.toDouble();
    final end = stop.toDouble();
    if (delta > 0) {
      while (current < end) {
        result.add(current);
        current += delta;
      }
    } else {
      while (current > end) {
        result.add(current);
        current += delta;
      }
    }
    return Vector64(result);
  }

  factory Vector64.linspace(num start, num stop, int count) {
    if (count < 0) {
      throw ArgumentError.value(count, 'count', 'Count must be non-negative.');
    }
    if (count == 0) {
      return Vector64.zeros(0);
    }
    if (count == 1) {
      return Vector64([start]);
    }
    final first = start.toDouble();
    final step = (stop.toDouble() - first) / (count - 1);
    return Vector64.storage(
      Float64List.fromList(
        List<double>.generate(count, (index) => first + index * step),
      ),
      copy: false,
    );
  }

  factory Vector64.fromJson(Map<String, dynamic> json) =>
      Vector64((json['data'] as List).cast<num>());

  Float64List get values => Float64List.fromList(_data);

  Float64List get unsafeValuesView => _data;

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
    if (other is Matrix64) {
      return multiplyMatrix(other);
    }
    return _binary(other, ElementwiseOp.multiply);
  }

  dynamic operator /(Object other) => _binary(other, ElementwiseOp.divide);

  double dot(Vector64 other) {
    _checkLength(other);
    return Kernels.dot(_data, 0, other._data, 0, length);
  }

  Vector64 multiplyMatrix(Matrix64 matrix) {
    if (length != matrix.rows) {
      throw ArgumentError(
        'Vector64/matrix shape mismatch: $length x '
        '(${matrix.rows}, ${matrix.columns}).',
      );
    }
    final matrixData = matrix.unsafeTransposedValuesView;
    final result = Kernels.mv(matrixData, matrix.columns, matrix.rows, _data);
    return Vector64.storage(result, copy: false);
  }

  double get sum => Kernels.sum(_data);

  double get mean {
    if (isEmpty) {
      throw StateError('Cannot compute the mean of an empty vector.');
    }
    return sum / length;
  }

  double median() {
    if (isEmpty) {
      throw StateError('Cannot compute the median of an empty vector.');
    }
    final sorted = toList(growable: false)..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
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
    VectorNorm.manhattan => fold<double>(0, (sum, value) => sum + value.abs()),
  };

  double prod() {
    if (isEmpty) {
      throw StateError('Cannot compute the product of an empty vector.');
    }
    var result = 1.0;
    for (final value in _data) {
      result *= value;
    }
    return result;
  }

  double maxValue() {
    if (isEmpty) {
      throw StateError('Cannot compute max of an empty vector.');
    }
    var result = _data.first;
    for (var i = 1; i < _data.length; i++) {
      if (_data[i] > result) {
        result = _data[i];
      }
    }
    return result;
  }

  double minValue() {
    if (isEmpty) {
      throw StateError('Cannot compute min of an empty vector.');
    }
    var result = _data.first;
    for (var i = 1; i < _data.length; i++) {
      if (_data[i] < result) {
        result = _data[i];
      }
    }
    return result;
  }

  double distanceTo(Vector64 other, [Distance distance = Distance.euclidean]) {
    _checkLength(other);
    final difference = this - other as Vector64;
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

  double cosine(Vector64 other) {
    _checkLength(other);
    final denominator = norm() * other.norm();
    if (denominator == 0) {
      throw StateError('Cannot compute cosine with a zero vector.');
    }
    return dot(other) / denominator;
  }

  Vector64 normalize([VectorNorm norm = VectorNorm.euclidean]) {
    final value = normType(norm);
    if (value == 0) {
      throw StateError('Cannot normalize a zero vector.');
    }
    return this / value;
  }

  Vector64 rescale() {
    final min = minValue();
    final max = maxValue();
    final span = max - min;
    if (span == 0) {
      return Vector64.zeros(length);
    }
    return mapValues((value) => (value - min) / span);
  }

  Vector64 unique() {
    final seen = <double>{};
    final result = <double>[];
    for (final value in _data) {
      if (seen.add(value)) {
        result.add(value);
      }
    }
    return Vector64(result);
  }

  Vector64 sample(Iterable<int> indices) {
    final result = <double>[];
    for (final index in indices) {
      RangeError.checkValidIndex(index, this, 'index', length);
      result.add(_data[index]);
    }
    return Vector64(result);
  }

  Vector64 subvector(int start, [int? end]) {
    final effectiveEnd = end ?? length;
    RangeError.checkValidRange(start, effectiveEnd, length);
    return Vector64.storage(
      Float64List.sublistView(_data, start, effectiveEnd),
    );
  }

  Vector64 filterElements(bool Function(double value, int index) predicate) {
    final result = <double>[];
    for (var i = 0; i < length; i++) {
      if (predicate(_data[i], i)) {
        result.add(_data[i]);
      }
    }
    return Vector64(result);
  }

  Vector64 pow(num exponent) =>
      mapValues((value) => math.pow(value, exponent).toDouble());

  Vector64 exp() => mapValues(math.exp);

  Vector64 log() => mapValues(math.log);

  Vector64 abs() => mapValues((value) => value.abs());

  Vector64 sqrt() => mapValues(math.sqrt);

  Vector64 mapValues(double Function(double value) transform) =>
      Vector64(_data.map(transform));

  Vector64 copy() => Vector64.storage(_data);

  Map<String, dynamic> toJson() => <String, dynamic>{
    'dtype': 'float64',
    'data': toList(growable: false),
  };

  dynamic _binary(Object other, ElementwiseOp op) {
    if (other is num) {
      return Vector64.storage(
        Kernels.scalar(_data, other.toDouble(), op),
        copy: false,
      );
    }
    if (other is Vector64) {
      _checkLength(other);
      return Vector64.storage(
        op == ElementwiseOp.add
            ? Kernels.add(_data, other._data)
            : Kernels.elementwise(_data, other._data, op),
        copy: false,
      );
    }
    if (other is Matrix64) {
      return _broadcastMatrix(other, op);
    }
    throw ArgumentError.value(
      other,
      'other',
      'Expected a num, Vector64, or Matrix64.',
    );
  }

  Matrix64 _broadcastMatrix(Matrix64 matrix, ElementwiseOp op) {
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
    return Matrix64.fromRows(rows);
  }

  void _checkLength(Vector64 other) {
    if (length != other.length) {
      throw ArgumentError(
        'Vector64 length mismatch: $length vs ${other.length}.',
      );
    }
  }

  double _hamming(Vector64 other) {
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

Vector64 vector(Iterable<num> values) => Vector64(values);

Vector64 vec(Iterable<num> values) => vector(values);

Vector64 vectorZeros(int length) => Vector64.zeros(length);

Vector64 vecZeros(int length) => vectorZeros(length);

Vector64 vectorOnes(int length) => Vector64.ones(length);

Vector64 vecOnes(int length) => vectorOnes(length);

Vector64 vectorFull(int length, num value) => Vector64.full(length, value);

Vector64 vecFull(int length, num value) => vectorFull(length, value);

Vector64 randomVector(int length, {int? seed}) =>
    Vector64.random(length, seed: seed);

Vector64 randVec(int length, {int? seed}) => randomVector(length, seed: seed);

Vector64 vectorRange(num start, num stop, {num step = 1}) =>
    Vector64.range(start, stop, step: step);

Vector64 vecRange(num start, num stop, {num step = 1}) =>
    vectorRange(start, stop, step: step);

Vector64 vectorLinspace(num start, num stop, int count) =>
    Vector64.linspace(start, stop, count);

Vector64 vecLinspace(num start, num stop, int count) =>
    vectorLinspace(start, stop, count);

typedef Vector = Vector64;
