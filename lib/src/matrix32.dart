import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'decompositions.dart';
import 'iterative.dart';
import 'kernels.dart';
import 'krylov.dart';
import 'matrix.dart';
import 'types.dart';
import 'vector.dart';
import 'vector32.dart';

class Matrix32 {
  final int rows;
  final int columns;
  final Float32List _data;
  Float32List? _transposeCache;
  int _transposeCacheVersion = -1;
  int _version = 0;

  factory Matrix32(Iterable<Iterable<num>> rows) => Matrix32.fromRows(rows);

  Matrix32.storage(
    this.rows,
    this.columns,
    Float32List data, {
    bool copy = true,
  }) : _data = copy ? Float32List.fromList(data) : data {
    if (rows < 0 || columns < 0) {
      throw ArgumentError('Shape values must be non-negative.');
    }
    if (data.length != rows * columns) {
      throw ArgumentError(
        'Storage length ${data.length} does not match shape ($rows, $columns).',
      );
    }
  }

  factory Matrix32.fromRows(Iterable<Iterable<num>> rows) {
    if (rows is List<List<double>>) {
      return _fromDoubleRows32(rows);
    }
    final rowList = rows is List<Iterable<num>>
        ? rows
        : rows.toList(growable: false);
    if (rowList.isEmpty) {
      return Matrix32.storage(0, 0, Float32List(0), copy: false);
    }
    final columns = _rowLength(rowList.first);
    final data = Float32List(rowList.length * columns);
    for (var row = 0; row < rowList.length; row++) {
      final offset = row * columns;
      _copyRow(rowList[row], data, offset, columns, row);
    }
    return Matrix32.storage(rowList.length, columns, data, copy: false);
  }

  factory Matrix32.fromFlat(Iterable<num> values, int rows, int columns) {
    final data = _flatValues32(values);
    return Matrix32.storage(rows, columns, data, copy: false);
  }

  factory Matrix32.fromColumns(Iterable<Iterable<num>> columns) =>
      Matrix32.fromRows(columns).transpose;

  factory Matrix32.fromMatrix(Matrix64 matrix) =>
      Matrix32.fromFlat(matrix.values, matrix.rows, matrix.columns);

  factory Matrix32.fromByteData(ByteData data, int rows, int columns) {
    final expectedBytes = rows * columns * Float32List.bytesPerElement;
    if (data.lengthInBytes != expectedBytes) {
      throw ArgumentError(
        'ByteData length ${data.lengthInBytes} does not match $expectedBytes '
        'bytes for shape ($rows, $columns).',
      );
    }
    return Matrix32.storage(
      rows,
      columns,
      data.buffer.asFloat32List(data.offsetInBytes, rows * columns),
    );
  }

  factory Matrix32.fromJson(Map<String, dynamic> json) {
    final rows = json['rows'] as int;
    final columns = json['columns'] as int;
    final values = (json['data'] as List).cast<num>();
    return Matrix32.fromFlat(values, rows, columns);
  }

  factory Matrix32.zeros(int rows, int columns) =>
      Matrix32.storage(rows, columns, Float32List(rows * columns), copy: false);

  factory Matrix32.zero(int rows, int columns) => Matrix32.zeros(rows, columns);

  factory Matrix32.ones(int rows, int columns) =>
      Matrix32.full(rows, columns, 1.0);

  factory Matrix32.one(int rows, int columns) => Matrix32.ones(rows, columns);

  factory Matrix32.full(int rows, int columns, num value) => Matrix32.storage(
    rows,
    columns,
    Kernels.filled32(rows * columns, value.toDouble()),
    copy: false,
  );

  factory Matrix32.number(num value, int rows, int columns) =>
      Matrix32.full(rows, columns, value);

  factory Matrix32.scalar(num value, int size) =>
      Matrix32.diag(List<num>.filled(size, value));

  factory Matrix32.random(int rows, int columns, {int? seed}) {
    final random = seed == null ? math.Random() : math.Random(seed);
    return Matrix32.storage(
      rows,
      columns,
      Float32List.fromList(
        List<double>.generate(rows * columns, (_) => random.nextDouble()),
      ),
      copy: false,
    );
  }

  factory Matrix32.randomSPD(int size, {int? seed, double diagonalShift = 1}) =>
      Matrix32.fromMatrix(
        Matrix64.randomSPD(size, seed: seed, diagonalShift: diagonalShift),
      );

  factory Matrix32.eye(int size) {
    final data = Float32List(size * size);
    for (var i = 0; i < size; i++) {
      data[i * size + i] = 1.0;
    }
    return Matrix32.storage(size, size, data, copy: false);
  }

  factory Matrix32.diag(Iterable<num> diagonal) {
    final values = diagonal
        .map((value) => value.toDouble())
        .toList(growable: false);
    final data = Float32List(values.length * values.length);
    for (var i = 0; i < values.length; i++) {
      data[i * values.length + i] = values[i];
    }
    return Matrix32.storage(values.length, values.length, data, copy: false);
  }

  factory Matrix32.arange(
    num start,
    num stop, {
    num step = 1,
    int columns = 1,
  }) {
    final values = Vector64.range(start, stop, step: step);
    if (values.length % columns != 0) {
      throw ArgumentError(
        'Generated ${values.length} values; cannot reshape into $columns '
        'columns.',
      );
    }
    return Matrix32.fromFlat(values, values.length ~/ columns, columns);
  }

  factory Matrix32.linspace(num start, num stop, int count, {int columns = 1}) {
    final values = Vector64.linspace(start, stop, count);
    if (values.length % columns != 0) {
      throw ArgumentError(
        'Generated ${values.length} values; cannot reshape into $columns '
        'columns.',
      );
    }
    return Matrix32.fromFlat(values, values.length ~/ columns, columns);
  }

  int get rowCount => rows;

  int get columnCount => columns;

  int get count => _data.length;

  (int, int) get shape => (rows, columns);

  bool get isSquare => rows == columns;

  Float32List get values => Float32List.fromList(_data);

  Float32List get unsafeValuesView => _data;

  Float32List get unsafeTransposedValuesView => _transposedData();

  double get max => toFloat64().max;

  double get min => toFloat64().min;

  double get sum => Kernels.sum32(_data);

  double get mean {
    if (count == 0) {
      throw StateError('Matrix64 is empty.');
    }
    return sum / count;
  }

  double get trace {
    if (!isSquare) {
      throw StateError('Expected a square matrix, got ($rows, $columns).');
    }
    var result = 0.0;
    for (var i = 0; i < rows; i++) {
      result += _data[i * columns + i];
    }
    return result;
  }

  double get determinant => toFloat64().determinant;

  int get rank => toFloat64().rank;

  Matrix32 get inverse => Matrix32.fromMatrix(toFloat64().inverse);

  Matrix32 get rowEchelonForm =>
      Matrix32.fromMatrix(toFloat64().rowEchelonForm);

  Matrix32 get transpose {
    final result = Kernels.transpose32(_data, rows, columns);
    return Matrix32.storage(columns, rows, result, copy: false);
  }

  Matrix32 get t => transpose;

  List<List<double>> toRows() => List<List<double>>.generate(
    rows,
    (row) => List<double>.generate(columns, (column) => get(row, column)),
    growable: false,
  );

  Matrix64 toFloat64() => Matrix64.fromFlat(_data, rows, columns);

  Map<String, dynamic> toJson() => <String, dynamic>{
    'dtype': 'float32',
    'rows': rows,
    'columns': columns,
    'data': values.toList(growable: false),
  };

  double get(int row, int column) => _data[_offset(row, column)];

  double call(int row, int column) => get(row, column);

  void set(int row, int column, num value) {
    _data[_offset(row, column)] = value.toDouble();
    _markMutated();
  }

  Matrix32Row operator [](int row) {
    _checkRow(row);
    return Matrix32Row._(this, row);
  }

  Vector64 row(int row) => toFloat64().row(row);

  Vector64 column(int column) => toFloat64().column(column);

  Vector64 col(int column) => this.column(column);

  void setRow(int row, Iterable<num> values) {
    _checkRow(row);
    final rowValues = values.toList(growable: false);
    if (rowValues.length != columns) {
      throw ArgumentError(
        'Row length ${rowValues.length} does not match $columns columns.',
      );
    }
    for (var column = 0; column < columns; column++) {
      _data[row * columns + column] = rowValues[column].toDouble();
    }
    _markMutated();
  }

  void setColumn(int column, Iterable<num> values) {
    _checkColumn(column);
    final columnValues = values.toList(growable: false);
    if (columnValues.length != rows) {
      throw ArgumentError(
        'Column length ${columnValues.length} does not match $rows rows.',
      );
    }
    for (var row = 0; row < rows; row++) {
      _data[row * columns + column] = columnValues[row].toDouble();
    }
    _markMutated();
  }

  Matrix32 reshape(int rows, int columns) =>
      Matrix32.fromMatrix(toFloat64().reshape(rows, columns));

  Matrix32 slice({
    int rowStart = 0,
    int? rowEnd,
    int columnStart = 0,
    int? columnEnd,
  }) => Matrix32.fromMatrix(
    toFloat64().slice(
      rowStart: rowStart,
      rowEnd: rowEnd,
      columnStart: columnStart,
      columnEnd: columnEnd,
    ),
  );

  Matrix32 copy() => Matrix32.storage(rows, columns, _data);

  List<double> flatten() => values.toList(growable: false);

  Vector64 toVector() => toFloat64().toVector();

  Matrix32 mapValues(double Function(double value) transform) =>
      Matrix32.fromFlat(_data.map(transform), rows, columns);

  Matrix32 mapElements(double Function(double value) transform) =>
      mapValues(transform);

  Matrix32 mapRows(Vector64 Function(Vector64 row) transform) =>
      Matrix32.fromMatrix(toFloat64().mapRows(transform));

  Matrix32 mapColumns(Vector64 Function(Vector64 column) transform) =>
      Matrix32.fromMatrix(toFloat64().mapColumns(transform));

  Matrix32 filterRows(bool Function(Vector64 row, int index) predicate) =>
      Matrix32.fromMatrix(toFloat64().filterRows(predicate));

  Matrix32 filterColumns(bool Function(Vector64 column, int index) predicate) =>
      Matrix32.fromMatrix(toFloat64().filterColumns(predicate));

  Vector64 reduceRows(
    Vector64 Function(Vector64 previous, Vector64 current) combine,
  ) => toFloat64().reduceRows(combine);

  Vector64 reduceColumns(
    Vector64 Function(Vector64 previous, Vector64 current) combine,
  ) => toFloat64().reduceColumns(combine);

  Matrix32 sample({Iterable<int>? rowIndices, Iterable<int>? columnIndices}) =>
      Matrix32.fromMatrix(
        toFloat64().sample(
          rowIndices: rowIndices,
          columnIndices: columnIndices,
        ),
      );

  Matrix32 uniqueRows() => Matrix32.fromMatrix(toFloat64().uniqueRows());

  Matrix32 sort(
    double Function(Vector64 vector) selectSortValue, {
    Axis axis = Axis.rows,
    SortDirection direction = SortDirection.asc,
  }) => Matrix32.fromMatrix(
    toFloat64().sort(selectSortValue, axis: axis, direction: direction),
  );

  Vector64 meanByAxis([Axis axis = Axis.columns]) =>
      toFloat64().meanByAxis(axis);

  Vector64 variance([Axis axis = Axis.columns]) => toFloat64().variance(axis);

  Vector64 deviation([Axis axis = Axis.columns]) => toFloat64().deviation(axis);

  Matrix32 operator +(Object other) => _binary(other, ElementwiseOp.add);

  Matrix32 operator -(Object other) => _binary(other, ElementwiseOp.subtract);

  dynamic operator *(Object other) {
    if (other is Matrix32) {
      return matmul(other);
    }
    if (other is Vector32) {
      return mv32(other);
    }
    if (other is Vector64) {
      return mv(other);
    }
    if (other is num) {
      return _binary(other, ElementwiseOp.multiply);
    }
    throw ArgumentError.value(
      other,
      'other',
      'Expected a num, Vector64, Vector32, or Matrix32.',
    );
  }

  Matrix32 operator /(Object other) => _binary(other, ElementwiseOp.divide);

  Matrix32 hadamard(Object other) => _binary(other, ElementwiseOp.multiply);

  /// Scalar multiply that preserves the static [Matrix32] type.
  ///
  /// `operator *` returns `dynamic` because it also dispatches to
  /// matrix/vector products; use [scale] when the result must stay
  /// statically typed for chaining.
  Matrix32 scale(num factor) => _binary(factor, ElementwiseOp.multiply);

  /// Add [vector] to every line along [axis] with an explicit
  /// orientation. `Axis.columns` expects `vector.length == columns`,
  /// `Axis.rows` expects `vector.length == rows` — removing the
  /// square-matrix ambiguity of the broadcasting operators.
  Matrix32 addVector(Vector32 vector, Axis axis) =>
      _broadcastAxis32(vector, axis, ElementwiseOp.add);

  Matrix32 subtractVector(Vector32 vector, Axis axis) =>
      _broadcastAxis32(vector, axis, ElementwiseOp.subtract);

  Matrix32 multiplyVector(Vector32 vector, Axis axis) =>
      _broadcastAxis32(vector, axis, ElementwiseOp.multiply);

  Matrix32 divideVector(Vector32 vector, Axis axis) =>
      _broadcastAxis32(vector, axis, ElementwiseOp.divide);

  Matrix32 vstack(Matrix32 other) =>
      Matrix32.fromMatrix(toFloat64().vstack(other.toFloat64()));

  Matrix32 hstack(Matrix32 other) =>
      Matrix32.fromMatrix(toFloat64().hstack(other.toFloat64()));

  Matrix32 augment(Matrix32 other) => hstack(other);

  Vector64 mv(Vector64 vector) {
    if (columns != vector.length) {
      throw ArgumentError(
        'Vector64 length ${vector.length} does not match matrix columns $columns.',
      );
    }
    final result = Float64List(rows);
    for (var row = 0; row < rows; row++) {
      var sum = 0.0;
      final offset = row * columns;
      for (var column = 0; column < columns; column++) {
        sum += _data[offset + column] * vector[column];
      }
      result[row] = sum;
    }
    return Vector64.storage(result, copy: false);
  }

  Vector32 mv32(Vector32 vector) {
    if (columns != vector.length) {
      throw ArgumentError(
        'Matrix64/vector shape mismatch: ($rows, $columns) x ${vector.length}.',
      );
    }
    final result = Kernels.mv32(_data, rows, columns, vector.unsafeValuesView);
    return Vector32.storage(result, copy: false);
  }

  double norm({double p = 2}) {
    if (p <= 0) {
      throw ArgumentError.value(p, 'p', 'Norm order must be positive.');
    }
    if (p == 2) {
      return math.sqrt(Kernels.dot32(_data, 0, _data, 0, count));
    }
    var total = 0.0;
    for (final value in _data) {
      total += math.pow(value.abs(), p).toDouble();
    }
    return math.pow(total, 1 / p).toDouble();
  }

  double prod() {
    if (count == 0) {
      throw StateError('Matrix64 is empty.');
    }
    var result = 1.0;
    for (final value in _data) {
      result *= value;
    }
    return result;
  }

  Matrix32 pow(num exponent) => Matrix32.fromMatrix(toFloat64().pow(exponent));

  Matrix32 exp() => Matrix32.fromMatrix(toFloat64().exp());

  Matrix32 log() => Matrix32.fromMatrix(toFloat64().log());

  Matrix32 abs() => Matrix32.fromMatrix(toFloat64().abs());

  Matrix32 sqrt() => Matrix32.fromMatrix(toFloat64().sqrt());

  double det({double tolerance = Matrix64.defaultTolerance}) =>
      toFloat64().det(tolerance: tolerance);

  Matrix32 inv({double tolerance = Matrix64.defaultTolerance}) =>
      Matrix32.fromMatrix(toFloat64().inv(tolerance: tolerance));

  Matrix32 solve(
    Matrix32 rhs, {
    double tolerance = Matrix64.defaultTolerance,
  }) => Matrix32.fromMatrix(
    toFloat64().solve(rhs.toFloat64(), tolerance: tolerance),
  );

  Matrix32 rref({double tolerance = Matrix64.defaultTolerance}) =>
      Matrix32.fromMatrix(toFloat64().rref(tolerance: tolerance));

  LUDecomposition lu({double tolerance = Matrix64.defaultTolerance}) =>
      toFloat64().lu(tolerance: tolerance);

  QRDecomposition qr({double tolerance = Matrix64.defaultTolerance}) =>
      toFloat64().qr(tolerance: tolerance);

  CholeskyDecomposition cholesky({
    double tolerance = Matrix64.defaultTolerance,
  }) => toFloat64().cholesky(tolerance: tolerance);

  IterativeResult jacobi(
    Vector64 b, {
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => toFloat64().jacobi(
    b,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  IterativeResult gaussSeidel(
    Vector64 b, {
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => toFloat64().gaussSeidel(
    b,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  IterativeResult sor(
    Vector64 b, {
    double omega = 1.0,
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => toFloat64().sor(
    b,
    omega: omega,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  KrylovResult conjugateGradient(
    Vector64 b, {
    Vector64? initialGuess,
    int? maxIterations,
    double tolerance = 1e-10,
  }) => toFloat64().conjugateGradient(
    b,
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  KrylovResult gmres(
    Vector64 b, {
    Vector64? initialGuess,
    int restart = 30,
    int maxIterations = 300,
    double tolerance = 1e-10,
  }) => toFloat64().gmres(
    b,
    initialGuess: initialGuess,
    restart: restart,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  ArnoldiResult arnoldi(
    Vector64 start,
    int dimension, {
    double tolerance = Matrix64.defaultTolerance,
  }) => toFloat64().arnoldi(start, dimension, tolerance: tolerance);

  PowerIterationResult powerIteration({
    Vector64? initialGuess,
    int maxIterations = 1000,
    double tolerance = 1e-10,
  }) => toFloat64().powerIteration(
    initialGuess: initialGuess,
    maxIterations: maxIterations,
    tolerance: tolerance,
  );

  Matrix32 matmul(Matrix32 other) {
    if (columns != other.rows) {
      throw ArgumentError(
        'Matrix64 multiplication shape mismatch: ($rows, $columns) x '
        '(${other.rows}, ${other.columns}).',
      );
    }
    final rightT = other._transposedData();
    final result = Float32List(rows * other.columns);
    final outColumns = other.columns;
    if (columns % 4 == 0) {
      final leftSimd = _data.buffer.asFloat32x4List();
      final rightSimd = rightT.buffer.asFloat32x4List();
      final simdColumns = columns ~/ 4;
      for (var row = 0; row < rows; row++) {
        final leftOffset = (row * columns) ~/ 4;
        final outOffset = row * outColumns;
        var col = 0;
        for (; col + 3 < outColumns; col += 4) {
          final rightOffset0 = col * simdColumns;
          final rightOffset1 = rightOffset0 + simdColumns;
          final rightOffset2 = rightOffset1 + simdColumns;
          final rightOffset3 = rightOffset2 + simdColumns;
          var lanes0 = Float32x4.zero();
          var lanes1 = Float32x4.zero();
          var lanes2 = Float32x4.zero();
          var lanes3 = Float32x4.zero();
          for (var k = 0; k < simdColumns; k++) {
            final left = leftSimd[leftOffset + k];
            lanes0 += left * rightSimd[rightOffset0 + k];
            lanes1 += left * rightSimd[rightOffset1 + k];
            lanes2 += left * rightSimd[rightOffset2 + k];
            lanes3 += left * rightSimd[rightOffset3 + k];
          }
          result[outOffset + col] = _sumLane(lanes0);
          result[outOffset + col + 1] = _sumLane(lanes1);
          result[outOffset + col + 2] = _sumLane(lanes2);
          result[outOffset + col + 3] = _sumLane(lanes3);
        }
        for (; col < outColumns; col++) {
          final rightOffset = col * simdColumns;
          var lanes = Float32x4.zero();
          for (var k = 0; k < simdColumns; k++) {
            lanes += leftSimd[leftOffset + k] * rightSimd[rightOffset + k];
          }
          result[outOffset + col] = _sumLane(lanes);
        }
      }
      return Matrix32.storage(rows, outColumns, result, copy: false);
    }

    final paddedColumns = ((columns + 3) ~/ 4) * 4;
    final simdColumns = paddedColumns ~/ 4;
    final leftPadded = Float32List(rows * paddedColumns);
    for (var row = 0; row < rows; row++) {
      final sourceOffset = row * columns;
      final targetOffset = row * paddedColumns;
      for (var k = 0; k < columns; k++) {
        leftPadded[targetOffset + k] = _data[sourceOffset + k];
      }
    }
    final rightPadded = Float32List(outColumns * paddedColumns);
    for (var col = 0; col < outColumns; col++) {
      final sourceOffset = col * columns;
      final targetOffset = col * paddedColumns;
      for (var k = 0; k < columns; k++) {
        rightPadded[targetOffset + k] = rightT[sourceOffset + k];
      }
    }
    final leftSimd = leftPadded.buffer.asFloat32x4List();
    final rightSimd = rightPadded.buffer.asFloat32x4List();
    for (var row = 0; row < rows; row++) {
      final leftOffset = row * simdColumns;
      final outOffset = row * outColumns;
      var col = 0;
      for (; col + 3 < outColumns; col += 4) {
        final rightOffset0 = col * simdColumns;
        final rightOffset1 = rightOffset0 + simdColumns;
        final rightOffset2 = rightOffset1 + simdColumns;
        final rightOffset3 = rightOffset2 + simdColumns;
        var lanes0 = Float32x4.zero();
        var lanes1 = Float32x4.zero();
        var lanes2 = Float32x4.zero();
        var lanes3 = Float32x4.zero();
        for (var k = 0; k < simdColumns; k++) {
          final left = leftSimd[leftOffset + k];
          lanes0 += left * rightSimd[rightOffset0 + k];
          lanes1 += left * rightSimd[rightOffset1 + k];
          lanes2 += left * rightSimd[rightOffset2 + k];
          lanes3 += left * rightSimd[rightOffset3 + k];
        }
        result[outOffset + col] = _sumLane(lanes0);
        result[outOffset + col + 1] = _sumLane(lanes1);
        result[outOffset + col + 2] = _sumLane(lanes2);
        result[outOffset + col + 3] = _sumLane(lanes3);
      }
      for (; col < outColumns; col++) {
        final rightOffset = col * simdColumns;
        var lanes = Float32x4.zero();
        for (var k = 0; k < simdColumns; k++) {
          lanes += leftSimd[leftOffset + k] * rightSimd[rightOffset + k];
        }
        result[outOffset + col] = _sumLane(lanes);
      }
    }
    return Matrix32.storage(rows, outColumns, result, copy: false);
  }

  Matrix32 _binary(Object other, ElementwiseOp op) {
    if (other is num) {
      return Matrix32.storage(
        rows,
        columns,
        Kernels.scalar32(_data, other.toDouble(), op),
        copy: false,
      );
    }
    if (other is Matrix32) {
      if (rows != other.rows || columns != other.columns) {
        throw ArgumentError(
          'Shape mismatch: ($rows, $columns) vs '
          '(${other.rows}, ${other.columns}).',
        );
      }
      return Matrix32.storage(
        rows,
        columns,
        op == ElementwiseOp.add
            ? Kernels.add32(_data, other._data)
            : Kernels.elementwise32(_data, other._data, op),
        copy: false,
      );
    }
    if (other is Vector32) {
      return _broadcastVector32(other, op);
    }
    if (other is Vector64) {
      return _broadcastVector32(Vector32.fromVector(other), op);
    }
    throw ArgumentError.value(
      other,
      'other',
      'Expected a num, Matrix32, or Vector32.',
    );
  }

  Matrix32 _broadcastAxis32(Vector32 vector, Axis axis, ElementwiseOp op) {
    final expected = axis == Axis.columns ? columns : rows;
    if (vector.length != expected) {
      throw ArgumentError(
        'Vector32 length ${vector.length} must equal '
        '${axis == Axis.columns ? 'columns $columns' : 'rows $rows'} '
        'to broadcast along $axis.',
      );
    }
    final vectorData = vector.unsafeValuesView;
    final result = Float32List(count);
    for (var row = 0; row < rows; row++) {
      final offset = row * columns;
      for (var column = 0; column < columns; column++) {
        result[offset + column] = _apply(
          _data[offset + column],
          vectorData[axis == Axis.columns ? column : row],
          op,
        );
      }
    }
    return Matrix32.storage(rows, columns, result, copy: false);
  }

  Matrix32 _broadcastVector32(Vector32 vector, ElementwiseOp op) {
    final vectorData = vector.unsafeValuesView;
    final result = Float32List(count);
    if (vector.length == columns) {
      if (columns % 4 == 0) {
        final simdColumns = columns ~/ 4;
        final vectorSimd = vectorData.buffer.asFloat32x4List(0, simdColumns);
        final resultSimd = result.buffer.asFloat32x4List(0, count ~/ 4);
        final dataSimd = _data.buffer.asFloat32x4List(0, count ~/ 4);
        for (var row = 0; row < rows; row++) {
          final rowOffset = row * simdColumns;
          for (var column = 0; column < simdColumns; column++) {
            final index = rowOffset + column;
            resultSimd[index] = switch (op) {
              ElementwiseOp.add => dataSimd[index] + vectorSimd[column],
              ElementwiseOp.subtract => dataSimd[index] - vectorSimd[column],
              ElementwiseOp.multiply => dataSimd[index] * vectorSimd[column],
              ElementwiseOp.divide => dataSimd[index] / vectorSimd[column],
            };
          }
        }
      } else {
        for (var row = 0; row < rows; row++) {
          final offset = row * columns;
          for (var column = 0; column < columns; column++) {
            result[offset + column] = _apply(
              _data[offset + column],
              vectorData[column],
              op,
            );
          }
        }
      }
      return Matrix32.storage(rows, columns, result, copy: false);
    }
    if (vector.length == rows) {
      for (var row = 0; row < rows; row++) {
        final lane = Float32x4(
          vectorData[row],
          vectorData[row],
          vectorData[row],
          vectorData[row],
        );
        final offset = row * columns;
        var column = 0;
        for (; column + 3 < columns; column += 4) {
          final left = Float32x4(
            _data[offset + column],
            _data[offset + column + 1],
            _data[offset + column + 2],
            _data[offset + column + 3],
          );
          final applied = switch (op) {
            ElementwiseOp.add => left + lane,
            ElementwiseOp.subtract => left - lane,
            ElementwiseOp.multiply => left * lane,
            ElementwiseOp.divide => left / lane,
          };
          result[offset + column] = applied.x;
          result[offset + column + 1] = applied.y;
          result[offset + column + 2] = applied.z;
          result[offset + column + 3] = applied.w;
        }
        for (; column < columns; column++) {
          result[offset + column] = _apply(
            _data[offset + column],
            vectorData[row],
            op,
          );
        }
      }
      return Matrix32.storage(rows, columns, result, copy: false);
    }
    throw ArgumentError(
      'Vector64 length ${vector.length} must match matrix rows $rows or '
      'columns $columns for broadcasting.',
    );
  }

  Float32List _transposedData() {
    if (_transposeCacheVersion == _version && _transposeCache != null) {
      return _transposeCache!;
    }
    final transposed = Kernels.transpose32(_data, rows, columns);
    _transposeCache = transposed;
    _transposeCacheVersion = _version;
    return transposed;
  }

  void _markMutated() {
    _version++;
    _transposeCache = null;
    _transposeCacheVersion = -1;
  }

  int _offset(int row, int column) {
    _checkRow(row);
    _checkColumn(column);
    return row * columns + column;
  }

  void _checkRow(int row) {
    if (row < 0 || row >= rows) {
      throw RangeError.index(row, this, 'row', null, rows);
    }
  }

  void _checkColumn(int column) {
    if (column < 0 || column >= columns) {
      throw RangeError.index(column, this, 'column', null, columns);
    }
  }

  static double _sumLane(Float32x4 lane) => lane.x + lane.y + lane.z + lane.w;

  static double _apply(double left, double right, ElementwiseOp op) =>
      switch (op) {
        ElementwiseOp.add => left + right,
        ElementwiseOp.subtract => left - right,
        ElementwiseOp.multiply => left * right,
        ElementwiseOp.divide => left / right,
      };

  @override
  String toString() {
    final buffer = StringBuffer('Matrix32($rows, $columns)\n');
    for (var row = 0; row < rows; row++) {
      buffer.writeln(this[row]);
    }
    return buffer.toString().trimRight();
  }
}

List<num> _rowValues(Iterable<num> row) =>
    row is List<num> ? row : row.toList(growable: false);

int _rowLength(Iterable<num> row) => row is List<num> ? row.length : row.length;

void _copyRow(
  Iterable<num> row,
  Float32List target,
  int offset,
  int columns,
  int rowIndex,
) {
  if (row is List<double>) {
    if (row.length != columns) {
      throw ArgumentError(
        'Row $rowIndex has length ${row.length}; expected $columns.',
      );
    }
    for (var column = 0; column < columns; column++) {
      target[offset + column] = row[column];
    }
    return;
  }
  final values = _rowValues(row);
  if (values.length != columns) {
    throw ArgumentError(
      'Row $rowIndex has length ${values.length}; expected $columns.',
    );
  }
  for (var column = 0; column < columns; column++) {
    target[offset + column] = values[column].toDouble();
  }
}

Float32List _flatValues32(Iterable<num> values) {
  if (values is Float32List) {
    return Float32List.fromList(values);
  }
  final list = values is List<num> ? values : values.toList(growable: false);
  final data = Float32List(list.length);
  for (var i = 0; i < list.length; i++) {
    data[i] = list[i].toDouble();
  }
  return data;
}

class Matrix32Row extends ListBase<double> {
  final Matrix32 _matrix;
  final int _row;

  Matrix32Row._(this._matrix, this._row);

  @override
  int get length => _matrix.columns;

  @override
  set length(int newLength) {
    throw UnsupportedError('Matrix64 rows have a fixed length.');
  }

  @override
  double operator [](int index) => _matrix.get(_row, index);

  @override
  void operator []=(int index, double value) {
    _matrix.set(_row, index, value);
  }
}

Matrix32 matrix32(Iterable<Iterable<num>> values) => Matrix32.fromRows(values);

Matrix32 mat32(Iterable<Iterable<num>> values) => matrix32(values);

Matrix32 zeros32(int rows, int columns) => Matrix32.zeros(rows, columns);

Matrix32 ones32(int rows, int columns) => Matrix32.ones(rows, columns);

Matrix32 full32(int rows, int columns, num value) =>
    Matrix32.full(rows, columns, value);

Matrix32 rand32(int rows, int columns, {int? seed}) =>
    Matrix32.random(rows, columns, seed: seed);

Matrix32 eye32(int size) => Matrix32.eye(size);

Matrix32 diag32(Iterable<num> values) => Matrix32.diag(values);

Matrix32 arange32(num start, num stop, {num step = 1, int columns = 1}) =>
    Matrix32.arange(start, stop, step: step, columns: columns);

Matrix32 linspace32(num start, num stop, int count, {int columns = 1}) =>
    Matrix32.linspace(start, stop, count, columns: columns);

Matrix32 _fromDoubleRows32(List<List<double>> rows) {
  if (rows.isEmpty) {
    return Matrix32.storage(0, 0, Float32List(0), copy: false);
  }
  final columns = rows.first.length;
  final data = Float32List(rows.length * columns);
  for (var row = 0; row < rows.length; row++) {
    final values = rows[row];
    if (values.length != columns) {
      throw ArgumentError(
        'Row $row has length ${values.length}; expected $columns.',
      );
    }
    final offset = row * columns;
    for (var column = 0; column < columns; column++) {
      data[offset + column] = values[column];
    }
  }
  return Matrix32.storage(rows.length, columns, data, copy: false);
}
