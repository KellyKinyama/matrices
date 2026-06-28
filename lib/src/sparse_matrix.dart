import 'dart:collection';
import 'dart:typed_data';

import 'matrix.dart';
import 'matrix32.dart';
import 'vector.dart';

class SparseEntry {
  final int row;
  final int column;
  final double value;

  SparseEntry(this.row, this.column, num value) : value = value.toDouble();
}

class SparseMatrix {
  final int rows;
  final int columns;
  final Float64List values;
  final Int32List columnIndices;
  final Int32List rowPointers;

  SparseMatrix._(
    this.rows,
    this.columns,
    this.values,
    this.columnIndices,
    this.rowPointers,
  );

  factory SparseMatrix.csr({
    required int rows,
    required int columns,
    required Iterable<num> values,
    required Iterable<int> columnIndices,
    required Iterable<int> rowPointers,
  }) {
    final data = Float64List.fromList(
      values.map((value) => value.toDouble()).toList(growable: false),
    );
    final colIdx = Int32List.fromList(columnIndices.toList(growable: false));
    final rowPtr = Int32List.fromList(rowPointers.toList(growable: false));
    _validateCsr(rows, columns, data, colIdx, rowPtr);
    return SparseMatrix._(rows, columns, data, colIdx, rowPtr);
  }

  factory SparseMatrix.zeros(int rows, int columns) {
    _checkShape(rows, columns);
    return SparseMatrix._(
      rows,
      columns,
      Float64List(0),
      Int32List(0),
      Int32List(rows + 1),
    );
  }

  factory SparseMatrix.fromRows(
    Iterable<Iterable<num>> rows, {
    double tolerance = 0.0,
  }) {
    final materialized = rows
        .map((row) => row.map((value) => value.toDouble()).toList())
        .toList(growable: false);
    if (materialized.isEmpty) {
      return SparseMatrix.zeros(0, 0);
    }
    final columnCount = materialized.first.length;
    final rowPointers = Int32List(materialized.length + 1);
    final values = <double>[];
    final columnIndices = <int>[];
    for (var row = 0; row < materialized.length; row++) {
      if (materialized[row].length != columnCount) {
        throw ArgumentError('All rows must have the same length.');
      }
      for (var column = 0; column < columnCount; column++) {
        final value = materialized[row][column];
        if (value.abs() > tolerance) {
          values.add(value);
          columnIndices.add(column);
        }
      }
      rowPointers[row + 1] = values.length;
    }
    return SparseMatrix.csr(
      rows: materialized.length,
      columns: columnCount,
      values: values,
      columnIndices: columnIndices,
      rowPointers: rowPointers,
    );
  }

  factory SparseMatrix.fromDense(Matrix64 matrix, {double tolerance = 0.0}) =>
      SparseMatrix.fromRows(matrix.toRows(), tolerance: tolerance);

  factory SparseMatrix.fromTriplets(
    int rows,
    int columns,
    Iterable<SparseEntry> entries, {
    double tolerance = 0.0,
  }) {
    _checkShape(rows, columns);
    final byPosition = SplayTreeMap<int, double>();
    for (final entry in entries) {
      if (entry.row < 0 ||
          entry.row >= rows ||
          entry.column < 0 ||
          entry.column >= columns) {
        throw RangeError('Sparse entry index is out of bounds.');
      }
      if (entry.value.abs() <= tolerance) {
        continue;
      }
      final key = entry.row * columns + entry.column;
      byPosition[key] = (byPosition[key] ?? 0) + entry.value;
    }

    final values = <double>[];
    final columnIndices = <int>[];
    final rowPointers = Int32List(rows + 1);
    var currentRow = 0;
    for (final entry in byPosition.entries) {
      final row = entry.key ~/ columns;
      final column = entry.key % columns;
      while (currentRow < row) {
        rowPointers[++currentRow] = values.length;
      }
      if (entry.value.abs() > tolerance) {
        values.add(entry.value);
        columnIndices.add(column);
      }
    }
    while (currentRow < rows) {
      rowPointers[++currentRow] = values.length;
    }

    return SparseMatrix.csr(
      rows: rows,
      columns: columns,
      values: values,
      columnIndices: columnIndices,
      rowPointers: rowPointers,
    );
  }

  factory SparseMatrix.fromJson(Map<String, dynamic> json) {
    return SparseMatrix.csr(
      rows: json['rows'] as int,
      columns: json['columns'] as int,
      values: (json['values'] as List).cast<num>(),
      columnIndices: (json['columnIndices'] as List).cast<int>(),
      rowPointers: (json['rowPointers'] as List).cast<int>(),
    );
  }

  int get shapeRows => rows;

  int get shapeColumns => columns;

  (int, int) get shape => (rows, columns);

  int get nnz => values.length;

  double get density => rows == 0 || columns == 0 ? 0 : nnz / (rows * columns);

  double call(int row, int column) => get(row, column);

  double get(int row, int column) {
    _checkIndex(row, rows, 'row');
    _checkIndex(column, columns, 'column');
    final start = rowPointers[row];
    final end = rowPointers[row + 1];
    for (var index = start; index < end; index++) {
      final currentColumn = columnIndices[index];
      if (currentColumn == column) {
        return values[index];
      }
      if (currentColumn > column) {
        break;
      }
    }
    return 0.0;
  }

  Vector64 row(int row) {
    _checkIndex(row, rows, 'row');
    final result = Float64List(columns);
    for (var index = rowPointers[row]; index < rowPointers[row + 1]; index++) {
      result[columnIndices[index]] = values[index];
    }
    return Vector64.storage(result, copy: false);
  }

  Vector64 column(int column) {
    _checkIndex(column, columns, 'column');
    final result = Float64List(rows);
    for (var row = 0; row < rows; row++) {
      result[row] = get(row, column);
    }
    return Vector64.storage(result, copy: false);
  }

  Matrix64 toDense() {
    final result = Float64List(rows * columns);
    for (var row = 0; row < rows; row++) {
      final offset = row * columns;
      for (
        var index = rowPointers[row];
        index < rowPointers[row + 1];
        index++
      ) {
        result[offset + columnIndices[index]] = values[index];
      }
    }
    return Matrix64.storage(rows, columns, result, copy: false);
  }

  Vector64 mv(Vector64 vector) {
    if (vector.length != columns) {
      throw ArgumentError(
        'Vector64 length ${vector.length} does not match matrix columns $columns.',
      );
    }
    final result = Float64List(rows);
    for (var row = 0; row < rows; row++) {
      var sum = 0.0;
      for (
        var index = rowPointers[row];
        index < rowPointers[row + 1];
        index++
      ) {
        sum += values[index] * vector[columnIndices[index]];
      }
      result[row] = sum;
    }
    return Vector64.storage(result, copy: false);
  }

  Matrix64 matmul(Matrix64 other) {
    if (columns != other.rows) {
      throw ArgumentError(
        'Matrix64 dimensions do not match: $rows x $columns cannot multiply '
        '${other.rows} x ${other.columns}.',
      );
    }
    final result = Float64List(rows * other.columns);
    for (var row = 0; row < rows; row++) {
      final outOffset = row * other.columns;
      for (
        var index = rowPointers[row];
        index < rowPointers[row + 1];
        index++
      ) {
        final leftValue = values[index];
        final leftColumn = columnIndices[index];
        for (var column = 0; column < other.columns; column++) {
          result[outOffset + column] += leftValue * other(leftColumn, column);
        }
      }
    }
    return Matrix64.storage(rows, other.columns, result, copy: false);
  }

  SparseMatrix transpose() {
    final counts = Int32List(columns);
    for (final column in columnIndices) {
      counts[column]++;
    }

    final transposedPointers = Int32List(columns + 1);
    for (var i = 0; i < columns; i++) {
      transposedPointers[i + 1] = transposedPointers[i] + counts[i];
    }

    final next = Int32List.fromList(transposedPointers);
    final transposedValues = Float64List(nnz);
    final transposedColumns = Int32List(nnz);
    for (var row = 0; row < rows; row++) {
      for (
        var index = rowPointers[row];
        index < rowPointers[row + 1];
        index++
      ) {
        final column = columnIndices[index];
        final target = next[column]++;
        transposedValues[target] = values[index];
        transposedColumns[target] = row;
      }
    }

    return SparseMatrix._(
      columns,
      rows,
      transposedValues,
      transposedColumns,
      transposedPointers,
    );
  }

  SparseMatrix copy() => SparseMatrix._(
    rows,
    columns,
    Float64List.fromList(values),
    Int32List.fromList(columnIndices),
    Int32List.fromList(rowPointers),
  );

  SparseMatrix scale(num scalar) {
    final factor = scalar.toDouble();
    return SparseMatrix._(
      rows,
      columns,
      Float64List.fromList(values.map((value) => value * factor).toList()),
      Int32List.fromList(columnIndices),
      Int32List.fromList(rowPointers),
    );
  }

  Object operator *(Object other) {
    if (other is num) {
      return scale(other);
    }
    if (other is Vector64) {
      return mv(other);
    }
    if (other is Matrix64) {
      return matmul(other);
    }
    if (other is SparseMatrix) {
      return SparseMatrix.fromDense(matmul(other.toDense()));
    }
    throw ArgumentError(
      'Unsupported multiplication operand: ${other.runtimeType}.',
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
    'type': 'SparseMatrix',
    'rows': rows,
    'columns': columns,
    'values': values.toList(growable: false),
    'columnIndices': columnIndices.toList(growable: false),
    'rowPointers': rowPointers.toList(growable: false),
  };

  @override
  String toString() => 'SparseMatrix($rows x $columns, nnz: $nnz)';

  static void _validateCsr(
    int rows,
    int columns,
    Float64List values,
    Int32List columnIndices,
    Int32List rowPointers,
  ) {
    _checkShape(rows, columns);
    if (values.length != columnIndices.length) {
      throw ArgumentError(
        'values and columnIndices must have the same length.',
      );
    }
    if (rowPointers.length != rows + 1) {
      throw ArgumentError('rowPointers length must be rows + 1.');
    }
    if (rowPointers.isEmpty || rowPointers.first != 0) {
      throw ArgumentError('rowPointers must start at zero.');
    }
    if (rowPointers.last != values.length) {
      throw ArgumentError('rowPointers must end at values.length.');
    }
    for (var row = 0; row < rows; row++) {
      if (rowPointers[row] > rowPointers[row + 1]) {
        throw ArgumentError('rowPointers must be monotonic.');
      }
      var previousColumn = -1;
      for (
        var index = rowPointers[row];
        index < rowPointers[row + 1];
        index++
      ) {
        final column = columnIndices[index];
        if (column < 0 || column >= columns) {
          throw RangeError('columnIndices contains an out-of-bounds column.');
        }
        if (column <= previousColumn) {
          throw ArgumentError('CSR column indices must be sorted per row.');
        }
        previousColumn = column;
      }
    }
  }

  static void _checkShape(int rows, int columns) {
    if (rows < 0 || columns < 0) {
      throw RangeError('Matrix64 shape cannot be negative.');
    }
  }

  static void _checkIndex(int index, int size, String name) {
    if (index < 0 || index >= size) {
      throw RangeError.index(index, null, name, null, size);
    }
  }
}

extension MatrixSparseConversion on Matrix64 {
  SparseMatrix toSparse({double tolerance = 0.0}) =>
      SparseMatrix.fromDense(this, tolerance: tolerance);
}

extension Matrix32SparseConversion on Matrix32 {
  SparseMatrix toSparse({double tolerance = 0.0}) =>
      SparseMatrix.fromDense(toFloat64(), tolerance: tolerance);
}
