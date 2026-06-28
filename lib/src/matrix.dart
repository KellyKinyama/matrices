import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'kernels.dart';
import 'square_matrix.dart';
import 'types.dart';
import 'vector.dart';

class Matrix64 {
  static const double defaultTolerance = 1e-12;

  final int rows;
  final int columns;
  final Float64List _data;
  Float64List? _transposeCache;
  int _transposeCacheVersion = -1;
  int _version = 0;

  factory Matrix64([Object? source, int? columns]) {
    if (source == null) {
      return Matrix64.storage(0, 0, Float64List(0), copy: false);
    }
    if (source is int && columns != null) {
      return Matrix64.zeros(source, columns);
    }
    if (source is Matrix64) {
      return source.copy();
    }
    if (source is Iterable) {
      final values = source.toList(growable: false);
      if (values.isEmpty) {
        return Matrix64.storage(0, 0, Float64List(0), copy: false);
      }
      if (values.first is Iterable) {
        return Matrix64.fromRows(values.cast<Iterable<num>>());
      }
      if (values.first is num && columns != null) {
        if (columns <= 0) {
          throw ArgumentError.value(
            columns,
            'columns',
            'Column count must be positive.',
          );
        }
        if (values.length % columns != 0) {
          throw ArgumentError(
            'Flat data length ${values.length} is not divisible by $columns.',
          );
        }
        return Matrix64.fromFlat(
          values.cast<num>(),
          values.length ~/ columns,
          columns,
        );
      }
    }
    throw ArgumentError.value(
      source,
      'source',
      'Expected nested rows, another Matrix64, or flat values with columns.',
    );
  }

  Matrix64.storage(
    this.rows,
    this.columns,
    Float64List data, {
    bool copy = true,
  }) : _data = copy ? Float64List.fromList(data) : data {
    _checkShape(rows, columns);
    if (data.length != rows * columns) {
      throw ArgumentError(
        'Storage length ${data.length} does not match shape ($rows, $columns).',
      );
    }
  }

  factory Matrix64.fromRows(Iterable<Iterable<num>> rows) {
    if (rows is List<List<double>>) {
      return _fromDoubleRows(rows);
    }
    final rowList = rows is List<Iterable<num>>
        ? rows
        : rows.toList(growable: false);
    if (rowList.isEmpty) {
      return Matrix64.storage(0, 0, Float64List(0), copy: false);
    }
    final columns = _rowLength(rowList.first);
    final data = Float64List(rowList.length * columns);
    for (var row = 0; row < rowList.length; row++) {
      final offset = row * columns;
      _copyRow(rowList[row], data, offset, columns, row);
    }
    return Matrix64.storage(rowList.length, columns, data, copy: false);
  }

  factory Matrix64.fromList(List<List<num>> rows) => Matrix64.fromRows(rows);

  factory Matrix64.fromColumns(Iterable<Iterable<num>> columns) =>
      Matrix64.fromRows(columns).transpose;

  factory Matrix64.fromFlat(Iterable<num> values, int rows, int columns) {
    _checkShape(rows, columns);
    final data = _flatValues(values);
    if (data.length != rows * columns) {
      throw ArgumentError(
        'Flat data length ${data.length} does not match shape '
        '($rows, $columns).',
      );
    }
    return Matrix64.storage(rows, columns, data, copy: false);
  }

  factory Matrix64.fromFlattenedList(List<num> values, int rows, int columns) =>
      Matrix64.fromFlat(values, rows, columns);

  factory Matrix64.fromByteData(ByteData data, int rows, int columns) {
    _checkShape(rows, columns);
    final expectedBytes = rows * columns * Float64List.bytesPerElement;
    if (data.lengthInBytes != expectedBytes) {
      throw ArgumentError(
        'ByteData length ${data.lengthInBytes} does not match $expectedBytes '
        'bytes for shape ($rows, $columns).',
      );
    }
    return Matrix64.storage(
      rows,
      columns,
      data.buffer.asFloat64List(data.offsetInBytes, rows * columns),
    );
  }

  factory Matrix64.fromJson(Map<String, dynamic> json) {
    final rows = json['rows'] as int;
    final columns = json['columns'] as int;
    final values = (json['data'] as List).cast<num>();
    return Matrix64.fromFlat(values, rows, columns);
  }

  factory Matrix64.zeros(int rows, int columns) =>
      Matrix64.storage(rows, columns, Float64List(rows * columns), copy: false);

  factory Matrix64.zero(int rows, int columns) => Matrix64.zeros(rows, columns);

  factory Matrix64.ones(int rows, int columns) =>
      Matrix64.full(rows, columns, 1.0);

  factory Matrix64.one(int rows, int columns) => Matrix64.ones(rows, columns);

  factory Matrix64.full(int rows, int columns, num value) => Matrix64.storage(
    rows,
    columns,
    Kernels.filled(rows * columns, value.toDouble()),
    copy: false,
  );

  factory Matrix64.number(num value, int rows, int columns) =>
      Matrix64.full(rows, columns, value);

  factory Matrix64.scalar(num value, int size) =>
      Matrix64.diag(List<num>.filled(size, value));

  factory Matrix64.random(int rows, int columns, {int? seed}) {
    _checkShape(rows, columns);
    final random = seed == null ? math.Random() : math.Random(seed);
    return Matrix64.storage(
      rows,
      columns,
      Float64List.fromList(
        List<double>.generate(rows * columns, (_) => random.nextDouble()),
      ),
      copy: false,
    );
  }

  factory Matrix64.randomSPD(
    int size, {
    int? seed,
    double diagonalShift = 1.0,
  }) {
    final a = Matrix64.random(size, size, seed: seed);
    return a.transpose * a + Matrix64.eye(size) * diagonalShift;
  }

  factory Matrix64.eye(int size) {
    _checkShape(size, size);
    final data = Float64List(size * size);
    for (var i = 0; i < size; i++) {
      data[i * size + i] = 1.0;
    }
    return Matrix64.storage(size, size, data, copy: false);
  }

  factory Matrix64.diag(Iterable<num> diagonal) {
    final values = diagonal
        .map((value) => value.toDouble())
        .toList(growable: false);
    final data = Float64List(values.length * values.length);
    for (var i = 0; i < values.length; i++) {
      data[i * values.length + i] = values[i];
    }
    return Matrix64.storage(values.length, values.length, data, copy: false);
  }

  factory Matrix64.arange(
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
    return Matrix64.fromFlat(values, values.length ~/ columns, columns);
  }

  factory Matrix64.linspace(num start, num stop, int count, {int columns = 1}) {
    final values = Vector64.linspace(start, stop, count);
    if (values.length % columns != 0) {
      throw ArgumentError(
        'Generated ${values.length} values; cannot reshape into $columns '
        'columns.',
      );
    }
    return Matrix64.fromFlat(values, values.length ~/ columns, columns);
  }

  int get rowCount => rows;

  int get columnCount => columns;

  (int, int) get shape => (rows, columns);

  int get count => _data.length;

  bool get isSquare => rows == columns;

  Float64List get values => Float64List.fromList(_data);

  Float64List get unsafeValuesView => _data;

  Float64List get unsafeTransposedValuesView => _transposedData();

  List<List<double>> get matrix => toRows();

  double get max {
    _checkNotEmpty();
    var result = _data.first;
    for (var i = 1; i < _data.length; i++) {
      if (_data[i] > result) {
        result = _data[i];
      }
    }
    return result;
  }

  double get min {
    _checkNotEmpty();
    var result = _data.first;
    for (var i = 1; i < _data.length; i++) {
      if (_data[i] < result) {
        result = _data[i];
      }
    }
    return result;
  }

  double get sum => Kernels.sum(_data);

  double get mean {
    _checkNotEmpty();
    return sum / count;
  }

  double get trace {
    _checkSquare();
    var result = 0.0;
    for (var i = 0; i < rows; i++) {
      result += get(i, i);
    }
    return result;
  }

  double get determinant => det();

  Matrix64 get inverse => inv();

  Matrix64 get transpose {
    final result = Kernels.transpose(_data, rows, columns);
    return Matrix64.storage(columns, rows, result, copy: false);
  }

  Matrix64 get t => transpose;

  Matrix64 get rowEchelonForm => rref();

  int get rank => rref()._rankFromRref();

  SquareMatrix get square {
    final size = math.min(rows, columns);
    final data = Float64List(size * size);
    for (var r = 0; r < size; r++) {
      for (var c = 0; c < size; c++) {
        data[r * size + c] = get(r, c);
      }
    }
    return SquareMatrix.storage(size, data, copy: false);
  }

  double get(int row, int column) => _data[_offset(row, column)];

  void set(int row, int column, num value) {
    _data[_offset(row, column)] = value.toDouble();
    _markMutated();
  }

  double call(int row, int column) => get(row, column);

  MatrixRow operator [](int row) {
    _checkRow(row);
    return MatrixRow._(this, row);
  }

  void operator []=(int row, Iterable<num> values) => setRow(row, values);

  Vector64 row(int row) {
    _checkRow(row);
    return Vector64.storage(
      Float64List.sublistView(_data, row * columns, (row + 1) * columns),
    );
  }

  Vector64 column(int column) {
    _checkColumn(column);
    final result = Float64List(rows);
    for (var r = 0; r < rows; r++) {
      result[r] = get(r, column);
    }
    return Vector64.storage(result, copy: false);
  }

  Vector64 col(int column) => this.column(column);

  void setRow(int row, Iterable<num> values) {
    _checkRow(row);
    final rowValues = values.toList(growable: false);
    if (rowValues.length != columns) {
      throw ArgumentError(
        'Row length ${rowValues.length} does not match $columns columns.',
      );
    }
    final base = row * columns;
    for (var c = 0; c < columns; c++) {
      _data[base + c] = rowValues[c].toDouble();
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
    for (var r = 0; r < rows; r++) {
      set(r, column, columnValues[r]);
    }
    _markMutated();
  }

  Matrix64 reshape(int rows, int columns) {
    _checkShape(rows, columns);
    if (rows * columns != count) {
      throw ArgumentError(
        'Cannot reshape $count values into shape ($rows, $columns).',
      );
    }
    return Matrix64.storage(rows, columns, _data);
  }

  Matrix64 slice({
    int rowStart = 0,
    int? rowEnd,
    int columnStart = 0,
    int? columnEnd,
  }) {
    final endRow = rowEnd ?? rows;
    final endColumn = columnEnd ?? columns;
    if (rowStart < 0 ||
        columnStart < 0 ||
        endRow < rowStart ||
        endColumn < columnStart ||
        endRow > rows ||
        endColumn > columns) {
      throw RangeError('Invalid slice bounds.');
    }
    final outRows = endRow - rowStart;
    final outColumns = endColumn - columnStart;
    final result = Float64List(outRows * outColumns);
    for (var r = 0; r < outRows; r++) {
      for (var c = 0; c < outColumns; c++) {
        result[r * outColumns + c] = get(rowStart + r, columnStart + c);
      }
    }
    return Matrix64.storage(outRows, outColumns, result, copy: false);
  }

  Matrix64 copy() => Matrix64.storage(rows, columns, _data);

  List<double> flatten() => _data.toList(growable: false);

  Vector64 toVector() => Vector64.storage(values, copy: false);

  List<List<double>> toRows() => List<List<double>>.generate(
    rows,
    (row) => List<double>.generate(columns, (column) => get(row, column)),
    growable: false,
  );

  Map<String, dynamic> toJson() => <String, dynamic>{
    'dtype': 'float64',
    'rows': rows,
    'columns': columns,
    'data': flatten(),
  };

  Matrix64 mapValues(double Function(double value) transform) =>
      Matrix64.storage(
        rows,
        columns,
        Float64List.fromList(_data.map(transform).toList()),
      );

  Matrix64 mapElements(double Function(double value) transform) =>
      mapValues(transform);

  Matrix64 mapRows(Vector64 Function(Vector64 row) transform) {
    final mapped = <Vector64>[];
    for (var rowIndex = 0; rowIndex < rows; rowIndex++) {
      final next = transform(row(rowIndex));
      if (next.length != columns) {
        throw ArgumentError(
          'Mapped row length ${next.length} does not match $columns.',
        );
      }
      mapped.add(next);
    }
    return Matrix64.fromRows(mapped);
  }

  Matrix64 mapColumns(Vector64 Function(Vector64 column) transform) {
    final mapped = <Vector64>[];
    for (var columnIndex = 0; columnIndex < columns; columnIndex++) {
      final next = transform(column(columnIndex));
      if (next.length != rows) {
        throw ArgumentError(
          'Mapped column length ${next.length} does not match $rows.',
        );
      }
      mapped.add(next);
    }
    return Matrix64.fromColumns(mapped);
  }

  Matrix64 filterRows(bool Function(Vector64 row, int index) predicate) {
    final kept = <Vector64>[];
    for (var rowIndex = 0; rowIndex < rows; rowIndex++) {
      final current = row(rowIndex);
      if (predicate(current, rowIndex)) {
        kept.add(current);
      }
    }
    return Matrix64.fromRows(kept);
  }

  Matrix64 filterColumns(bool Function(Vector64 column, int index) predicate) {
    final kept = <Vector64>[];
    for (var columnIndex = 0; columnIndex < columns; columnIndex++) {
      final current = column(columnIndex);
      if (predicate(current, columnIndex)) {
        kept.add(current);
      }
    }
    return Matrix64.fromColumns(kept);
  }

  Vector64 reduceRows(
    Vector64 Function(Vector64 previous, Vector64 current) combine,
  ) {
    _checkNotEmpty();
    var accumulator = row(0);
    for (var rowIndex = 1; rowIndex < rows; rowIndex++) {
      accumulator = combine(accumulator, row(rowIndex));
    }
    return accumulator;
  }

  Vector64 reduceColumns(
    Vector64 Function(Vector64 previous, Vector64 current) combine,
  ) {
    _checkNotEmpty();
    var accumulator = column(0);
    for (var columnIndex = 1; columnIndex < columns; columnIndex++) {
      accumulator = combine(accumulator, column(columnIndex));
    }
    return accumulator;
  }

  Matrix64 sample({Iterable<int>? rowIndices, Iterable<int>? columnIndices}) {
    final selectedRows = (rowIndices ?? Iterable<int>.generate(rows)).toList();
    final selectedColumns = (columnIndices ?? Iterable<int>.generate(columns))
        .toList();
    for (final rowIndex in selectedRows) {
      _checkRow(rowIndex);
    }
    for (final columnIndex in selectedColumns) {
      _checkColumn(columnIndex);
    }
    return Matrix64.fromRows(
      selectedRows.map(
        (rowIndex) =>
            selectedColumns.map((columnIndex) => get(rowIndex, columnIndex)),
      ),
    );
  }

  Matrix64 uniqueRows() {
    final seen = <String>{};
    final kept = <Vector64>[];
    for (var rowIndex = 0; rowIndex < rows; rowIndex++) {
      final current = row(rowIndex);
      final key = current.toList(growable: false).join('\u0001');
      if (seen.add(key)) {
        kept.add(current);
      }
    }
    return Matrix64.fromRows(kept);
  }

  Matrix64 sort(
    double Function(Vector64 vector) selectSortValue, {
    Axis axis = Axis.rows,
    SortDirection direction = SortDirection.asc,
  }) {
    final sign = direction == SortDirection.asc ? 1 : -1;
    if (axis == Axis.rows) {
      final sorted = List<Vector64>.generate(rows, row)
        ..sort(
          (a, b) => sign * selectSortValue(a).compareTo(selectSortValue(b)),
        );
      return Matrix64.fromRows(sorted);
    }
    final sorted = List<Vector64>.generate(columns, column)
      ..sort((a, b) => sign * selectSortValue(a).compareTo(selectSortValue(b)));
    return Matrix64.fromColumns(sorted);
  }

  Vector64 meanByAxis([Axis axis = Axis.columns]) {
    _checkNotEmpty();
    if (axis == Axis.columns) {
      return Vector64(
        List<double>.generate(
          columns,
          (columnIndex) => column(columnIndex).mean,
        ),
      );
    }
    return Vector64(
      List<double>.generate(rows, (rowIndex) => row(rowIndex).mean),
    );
  }

  Vector64 variance([Axis axis = Axis.columns]) {
    final means = meanByAxis(axis);
    if (axis == Axis.columns) {
      return Vector64(
        List<double>.generate(columns, (columnIndex) {
          final current = column(columnIndex);
          var total = 0.0;
          for (final value in current) {
            final diff = value - means[columnIndex];
            total += diff * diff;
          }
          return total / current.length;
        }),
      );
    }
    return Vector64(
      List<double>.generate(rows, (rowIndex) {
        final current = row(rowIndex);
        var total = 0.0;
        for (final value in current) {
          final diff = value - means[rowIndex];
          total += diff * diff;
        }
        return total / current.length;
      }),
    );
  }

  Vector64 deviation([Axis axis = Axis.columns]) =>
      variance(axis).mapValues(math.sqrt);

  double prod() {
    _checkNotEmpty();
    var result = 1.0;
    for (final value in _data) {
      result *= value;
    }
    return result;
  }

  Matrix64 pow(num exponent) =>
      mapValues((value) => math.pow(value, exponent).toDouble());

  Matrix64 exp() => mapValues(math.exp);

  Matrix64 log() => mapValues(math.log);

  Matrix64 abs() => mapValues((value) => value.abs());

  Matrix64 sqrt() => mapValues(math.sqrt);

  Matrix64 operator +(Object other) => _binary(other, ElementwiseOp.add);

  Matrix64 operator -(Object other) => _binary(other, ElementwiseOp.subtract);

  dynamic operator *(Object other) {
    if (other is Matrix64) {
      return matmul(other);
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
      'Expected a num, Vector64, or Matrix64.',
    );
  }

  Matrix64 operator /(Object other) => _binary(other, ElementwiseOp.divide);

  Matrix64 hadamard(Object other) => _binary(other, ElementwiseOp.multiply);

  Matrix64 matmul(Matrix64 other) {
    if (columns != other.rows) {
      throw ArgumentError(
        'Matrix64 multiplication shape mismatch: ($rows, $columns) x '
        '(${other.rows}, ${other.columns}).',
      );
    }
    final rightT = other._transposedData();
    final result = Float64List(rows * other.columns);
    final outColumns = other.columns;
    if (columns.isEven) {
      final leftSimd = _data.buffer.asFloat64x2List();
      final rightSimd = rightT.buffer.asFloat64x2List();
      final simdColumns = columns ~/ 2;
      for (var r = 0; r < rows; r++) {
        final leftOffset = (r * columns) ~/ 2;
        final outOffset = r * outColumns;
        var c = 0;
        for (; c + 3 < outColumns; c += 4) {
          final rightOffset0 = c * simdColumns;
          final rightOffset1 = rightOffset0 + simdColumns;
          final rightOffset2 = rightOffset1 + simdColumns;
          final rightOffset3 = rightOffset2 + simdColumns;
          var lanes0 = Float64x2.zero();
          var lanes1 = Float64x2.zero();
          var lanes2 = Float64x2.zero();
          var lanes3 = Float64x2.zero();
          for (var k = 0; k < simdColumns; k++) {
            final left = leftSimd[leftOffset + k];
            lanes0 += left * rightSimd[rightOffset0 + k];
            lanes1 += left * rightSimd[rightOffset1 + k];
            lanes2 += left * rightSimd[rightOffset2 + k];
            lanes3 += left * rightSimd[rightOffset3 + k];
          }
          result[outOffset + c] = lanes0.x + lanes0.y;
          result[outOffset + c + 1] = lanes1.x + lanes1.y;
          result[outOffset + c + 2] = lanes2.x + lanes2.y;
          result[outOffset + c + 3] = lanes3.x + lanes3.y;
        }
        for (; c < outColumns; c++) {
          final rightOffset = c * simdColumns;
          var lanes = Float64x2.zero();
          var k = 0;
          for (; k < simdColumns; k++) {
            lanes += leftSimd[leftOffset + k] * rightSimd[rightOffset + k];
          }
          result[outOffset + c] = lanes.x + lanes.y;
        }
      }
      return Matrix64.storage(rows, outColumns, result, copy: false);
    }
    for (var r = 0; r < rows; r++) {
      final leftOffset = r * columns;
      final outOffset = r * outColumns;
      for (var c = 0; c < outColumns; c++) {
        final rightOffset = c * columns;
        var sum = 0.0;
        var k = 0;
        for (; k + 3 < columns; k += 4) {
          sum +=
              _data[leftOffset + k] * rightT[rightOffset + k] +
              _data[leftOffset + k + 1] * rightT[rightOffset + k + 1] +
              _data[leftOffset + k + 2] * rightT[rightOffset + k + 2] +
              _data[leftOffset + k + 3] * rightT[rightOffset + k + 3];
        }
        for (; k < columns; k++) {
          sum += _data[leftOffset + k] * rightT[rightOffset + k];
        }
        result[outOffset + c] = sum;
      }
    }
    return Matrix64.storage(rows, outColumns, result, copy: false);
  }

  Vector64 mv(Vector64 vector) {
    if (columns != vector.length) {
      throw ArgumentError(
        'Matrix64/vector shape mismatch: ($rows, $columns) x '
        '${vector.length}.',
      );
    }
    final result = Kernels.mv(_data, rows, columns, vector.unsafeValuesView);
    return Vector64.storage(result, copy: false);
  }

  Matrix64 vstack(Matrix64 other) {
    if (columns != other.columns) {
      throw ArgumentError('Column mismatch: $columns vs ${other.columns}.');
    }
    final result = Float64List(count + other.count);
    result.setRange(0, count, _data);
    result.setRange(count, result.length, other._data);
    return Matrix64.storage(rows + other.rows, columns, result, copy: false);
  }

  Matrix64 hstack(Matrix64 other) {
    if (rows != other.rows) {
      throw ArgumentError('Row mismatch: $rows vs ${other.rows}.');
    }
    final outColumns = columns + other.columns;
    final result = Float64List(rows * outColumns);
    for (var r = 0; r < rows; r++) {
      final outOffset = r * outColumns;
      result.setRange(outOffset, outOffset + columns, _data, r * columns);
      result.setRange(
        outOffset + columns,
        outOffset + outColumns,
        other._data,
        r * other.columns,
      );
    }
    return Matrix64.storage(rows, outColumns, result, copy: false);
  }

  Matrix64 augment(Matrix64 other) => hstack(other);

  double norm({double p = 2}) {
    if (p <= 0) {
      throw ArgumentError.value(p, 'p', 'Norm order must be positive.');
    }
    if (p == 2) {
      return math.sqrt(Kernels.dot(_data, 0, _data, 0, count));
    }
    var total = 0.0;
    for (final value in _data) {
      total += math.pow(value.abs(), p).toDouble();
    }
    return math.pow(total, 1 / p).toDouble();
  }

  double det({double tolerance = defaultTolerance}) {
    _checkSquare();
    if (rows == 0) {
      return 1.0;
    }
    final a = Float64List.fromList(_data);
    var sign = 1.0;
    var determinant = 1.0;
    for (var pivot = 0; pivot < rows; pivot++) {
      final pivotRow = _maxPivotRow(a, pivot, tolerance);
      final pivotValue = a[pivotRow * columns + pivot];
      if (pivotValue.abs() <= tolerance) {
        return 0.0;
      }
      if (pivotRow != pivot) {
        _swapRows(a, columns, pivot, pivotRow);
        sign = -sign;
      }
      final diagonal = a[pivot * columns + pivot];
      determinant *= diagonal;
      for (var r = pivot + 1; r < rows; r++) {
        final factor = a[r * columns + pivot] / diagonal;
        a[r * columns + pivot] = 0.0;
        for (var c = pivot + 1; c < columns; c++) {
          a[r * columns + c] -= factor * a[pivot * columns + c];
        }
      }
    }
    return determinant * sign;
  }

  Matrix64 inv({double tolerance = defaultTolerance}) {
    _checkSquare();
    return solve(Matrix64.eye(rows), tolerance: tolerance);
  }

  Matrix64 solve(Matrix64 rhs, {double tolerance = defaultTolerance}) {
    _checkSquare();
    if (rhs.rows != rows) {
      throw ArgumentError(
        'Right-hand side row count ${rhs.rows} does not match $rows.',
      );
    }
    final a = Float64List.fromList(_data);
    final b = Float64List.fromList(rhs._data);
    final n = rows;
    final m = rhs.columns;

    for (var pivot = 0; pivot < n; pivot++) {
      final pivotRow = _maxPivotRow(a, pivot, tolerance);
      if (a[pivotRow * n + pivot].abs() <= tolerance) {
        throw StateError('Matrix64 is singular to working precision.');
      }
      if (pivotRow != pivot) {
        _swapRows(a, n, pivot, pivotRow);
        _swapRows(b, m, pivot, pivotRow);
      }
      final diagonal = a[pivot * n + pivot];
      for (var r = pivot + 1; r < n; r++) {
        final factor = a[r * n + pivot] / diagonal;
        a[r * n + pivot] = 0.0;
        for (var c = pivot + 1; c < n; c++) {
          a[r * n + c] -= factor * a[pivot * n + c];
        }
        for (var c = 0; c < m; c++) {
          b[r * m + c] -= factor * b[pivot * m + c];
        }
      }
    }

    final x = Float64List(n * m);
    for (var c = 0; c < m; c++) {
      for (var r = n - 1; r >= 0; r--) {
        var total = b[r * m + c];
        for (var k = r + 1; k < n; k++) {
          total -= a[r * n + k] * x[k * m + c];
        }
        x[r * m + c] = total / a[r * n + r];
      }
    }
    return Matrix64.storage(n, m, x, copy: false);
  }

  Matrix64 rref({double tolerance = defaultTolerance}) {
    final result = Float64List.fromList(_data);
    var lead = 0;
    for (var r = 0; r < rows; r++) {
      if (lead >= columns) {
        break;
      }
      var pivot = r;
      while (result[pivot * columns + lead].abs() <= tolerance) {
        pivot++;
        if (pivot == rows) {
          pivot = r;
          lead++;
          if (lead == columns) {
            return Matrix64.storage(rows, columns, result, copy: false);
          }
        }
      }
      if (pivot != r) {
        _swapRows(result, columns, pivot, r);
      }
      final divisor = result[r * columns + lead];
      for (var c = 0; c < columns; c++) {
        result[r * columns + c] /= divisor;
      }
      for (var other = 0; other < rows; other++) {
        if (other == r) {
          continue;
        }
        final factor = result[other * columns + lead];
        if (factor.abs() <= tolerance) {
          continue;
        }
        for (var c = 0; c < columns; c++) {
          result[other * columns + c] -= factor * result[r * columns + c];
          if (result[other * columns + c].abs() <= tolerance) {
            result[other * columns + c] = 0.0;
          }
        }
      }
      lead++;
    }
    return Matrix64.storage(rows, columns, result, copy: false);
  }

  Matrix64 _binary(Object other, ElementwiseOp op) {
    if (other is num) {
      return Matrix64.storage(
        rows,
        columns,
        Kernels.scalar(_data, other.toDouble(), op),
        copy: false,
      );
    }
    if (other is Matrix64) {
      _checkSameShape(other);
      return Matrix64.storage(
        rows,
        columns,
        op == ElementwiseOp.add
            ? Kernels.add(_data, other._data)
            : Kernels.elementwise(_data, other._data, op),
        copy: false,
      );
    }
    if (other is Vector64) {
      return _broadcastVector(other, op);
    }
    throw ArgumentError.value(
      other,
      'other',
      'Expected a num, Matrix64, or Vector64.',
    );
  }

  Matrix64 _broadcastVector(Vector64 vector, ElementwiseOp op) {
    final vectorData = vector.values;
    final result = Float64List(count);
    if (vector.length == columns) {
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
      return Matrix64.storage(rows, columns, result, copy: false);
    }
    if (vector.length == rows) {
      for (var row = 0; row < rows; row++) {
        final offset = row * columns;
        for (var column = 0; column < columns; column++) {
          result[offset + column] = _apply(
            _data[offset + column],
            vectorData[row],
            op,
          );
        }
      }
      return Matrix64.storage(rows, columns, result, copy: false);
    }
    throw ArgumentError(
      'Vector64 length ${vector.length} must match matrix rows $rows or '
      'columns $columns for broadcasting.',
    );
  }

  static double _apply(double left, double right, ElementwiseOp op) =>
      switch (op) {
        ElementwiseOp.add => left + right,
        ElementwiseOp.subtract => left - right,
        ElementwiseOp.multiply => left * right,
        ElementwiseOp.divide => left / right,
      };

  int _rankFromRref({double tolerance = defaultTolerance}) {
    var rank = 0;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < columns; c++) {
        if (get(r, c).abs() > tolerance) {
          rank++;
          break;
        }
      }
    }
    return rank;
  }

  void _checkSameShape(Matrix64 other) {
    if (rows != other.rows || columns != other.columns) {
      throw ArgumentError(
        'Shape mismatch: ($rows, $columns) vs '
        '(${other.rows}, ${other.columns}).',
      );
    }
  }

  void _checkNotEmpty() {
    if (count == 0) {
      throw StateError('Matrix64 is empty.');
    }
  }

  void _checkSquare() {
    if (!isSquare) {
      throw StateError('Expected a square matrix, got ($rows, $columns).');
    }
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

  static void _checkShape(int rows, int columns) {
    if (rows < 0 || columns < 0) {
      throw ArgumentError('Shape values must be non-negative.');
    }
  }

  static int _maxPivotRow(Float64List data, int pivot, double tolerance) {
    final stride = math.sqrt(data.length).toInt();
    var pivotRow = pivot;
    var maxValue = data[pivot * stride + pivot].abs();
    for (var r = pivot + 1; r < stride; r++) {
      final value = data[r * stride + pivot].abs();
      if (value > maxValue) {
        maxValue = value;
        pivotRow = r;
      }
    }
    return pivotRow;
  }

  static void _swapRows(Float64List data, int stride, int a, int b) {
    for (var c = 0; c < stride; c++) {
      final tmp = data[a * stride + c];
      data[a * stride + c] = data[b * stride + c];
      data[b * stride + c] = tmp;
    }
  }

  Float64List _transposedData() {
    if (_transposeCacheVersion == _version && _transposeCache != null) {
      return _transposeCache!;
    }
    final transposed = Kernels.transpose(_data, rows, columns);
    _transposeCache = transposed;
    _transposeCacheVersion = _version;
    return transposed;
  }

  void _markMutated() {
    _version++;
    _transposeCache = null;
    _transposeCacheVersion = -1;
  }

  @override
  String toString() {
    final buffer = StringBuffer('Matrix64($rows, $columns)\n');
    for (var r = 0; r < rows; r++) {
      buffer.writeln(row(r));
    }
    return buffer.toString().trimRight();
  }
}

List<num> _rowValues(Iterable<num> row) =>
    row is List<num> ? row : row.toList(growable: false);

int _rowLength(Iterable<num> row) => row is List<num> ? row.length : row.length;

void _copyRow(
  Iterable<num> row,
  Float64List target,
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

Float64List _flatValues(Iterable<num> values) {
  if (values is Float64List) {
    return Float64List.fromList(values);
  }
  final list = values is List<num> ? values : values.toList(growable: false);
  final data = Float64List(list.length);
  for (var i = 0; i < list.length; i++) {
    data[i] = list[i].toDouble();
  }
  return data;
}

class MatrixRow extends ListBase<double> {
  final Matrix64 _matrix;
  final int _row;

  MatrixRow._(this._matrix, this._row);

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

Matrix64 matrix(Iterable<Iterable<num>> values) => Matrix64.fromRows(values);

Matrix64 mat(Iterable<Iterable<num>> values) => matrix(values);

Matrix64 zeros(int rows, int columns) => Matrix64.zeros(rows, columns);

Matrix64 ones(int rows, int columns) => Matrix64.ones(rows, columns);

Matrix64 full(int rows, int columns, num value) =>
    Matrix64.full(rows, columns, value);

Matrix64 rand(int rows, int columns, {int? seed}) =>
    Matrix64.random(rows, columns, seed: seed);

Matrix64 eye(int size) => Matrix64.eye(size);

Matrix64 diag(Iterable<num> diagonal) => Matrix64.diag(diagonal);

Matrix64 arange(num start, num stop, {num step = 1, int columns = 1}) =>
    Matrix64.arange(start, stop, step: step, columns: columns);

Matrix64 linspace(num start, num stop, int count, {int columns = 1}) =>
    Matrix64.linspace(start, stop, count, columns: columns);

typedef Matrix = Matrix64;

Matrix64 _fromDoubleRows(List<List<double>> rows) {
  if (rows.isEmpty) {
    return Matrix64.storage(0, 0, Float64List(0), copy: false);
  }
  final columns = rows.first.length;
  final data = Float64List(rows.length * columns);
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
  return Matrix64.storage(rows.length, columns, data, copy: false);
}
