import 'dart:typed_data';

enum ElementwiseOp { add, subtract, multiply, divide }

abstract final class Kernels {
  static Float64List add(Float64List left, Float64List right) {
    _checkSameLength(left, right);
    if (left.length <= 10000) {
      return _addScalarUnrolled(left, right);
    }
    final resultSimd = Float64x2List((left.length + 1) ~/ 2);
    final result = resultSimd.buffer.asFloat64List(0, left.length);
    final simdLength = left.length ~/ 2;
    if (simdLength > 0) {
      final leftSimd = left.buffer.asFloat64x2List(
        left.offsetInBytes,
        simdLength,
      );
      final rightSimd = right.buffer.asFloat64x2List(
        right.offsetInBytes,
        simdLength,
      );
      var i = 0;
      for (; i + 3 < simdLength; i += 4) {
        resultSimd[i] = leftSimd[i] + rightSimd[i];
        resultSimd[i + 1] = leftSimd[i + 1] + rightSimd[i + 1];
        resultSimd[i + 2] = leftSimd[i + 2] + rightSimd[i + 2];
        resultSimd[i + 3] = leftSimd[i + 3] + rightSimd[i + 3];
      }
      for (; i < simdLength; i++) {
        resultSimd[i] = leftSimd[i] + rightSimd[i];
      }
    }
    for (var i = simdLength * 2; i < left.length; i++) {
      result[i] = left[i] + right[i];
    }
    return result;
  }

  static Float64List _addScalarUnrolled(Float64List left, Float64List right) {
    final result = Float64List(left.length);
    var i = 0;
    for (; i + 7 < left.length; i += 8) {
      result[i] = left[i] + right[i];
      result[i + 1] = left[i + 1] + right[i + 1];
      result[i + 2] = left[i + 2] + right[i + 2];
      result[i + 3] = left[i + 3] + right[i + 3];
      result[i + 4] = left[i + 4] + right[i + 4];
      result[i + 5] = left[i + 5] + right[i + 5];
      result[i + 6] = left[i + 6] + right[i + 6];
      result[i + 7] = left[i + 7] + right[i + 7];
    }
    for (; i < left.length; i++) {
      result[i] = left[i] + right[i];
    }
    return result;
  }

  static Float32List add32(Float32List left, Float32List right) {
    _checkSameLength32(left, right);
    final resultSimd = Float32x4List((left.length + 3) ~/ 4);
    final result = resultSimd.buffer.asFloat32List(0, left.length);
    final simdLength = left.length ~/ 4;
    if (simdLength > 0) {
      final leftSimd = left.buffer.asFloat32x4List(
        left.offsetInBytes,
        simdLength,
      );
      final rightSimd = right.buffer.asFloat32x4List(
        right.offsetInBytes,
        simdLength,
      );
      var i = 0;
      for (; i + 3 < simdLength; i += 4) {
        resultSimd[i] = leftSimd[i] + rightSimd[i];
        resultSimd[i + 1] = leftSimd[i + 1] + rightSimd[i + 1];
        resultSimd[i + 2] = leftSimd[i + 2] + rightSimd[i + 2];
        resultSimd[i + 3] = leftSimd[i + 3] + rightSimd[i + 3];
      }
      for (; i < simdLength; i++) {
        resultSimd[i] = leftSimd[i] + rightSimd[i];
      }
    }
    for (var i = simdLength * 4; i < left.length; i++) {
      result[i] = left[i] + right[i];
    }
    return result;
  }

  static Float64List filled(int length, double value) {
    final resultSimd = Float64x2List((length + 1) ~/ 2);
    final result = resultSimd.buffer.asFloat64List(0, length);
    if (value == 0.0) {
      return result;
    }
    final simdLength = length ~/ 2;
    final lane = Float64x2.splat(value);
    for (var i = 0; i < simdLength; i++) {
      resultSimd[i] = lane;
    }
    if (simdLength * 2 < length) {
      result[simdLength * 2] = value;
    }
    return result;
  }

  static Float32List filled32(int length, double value) {
    final resultSimd = Float32x4List((length + 3) ~/ 4);
    final result = resultSimd.buffer.asFloat32List(0, length);
    if (value == 0.0) {
      return result;
    }
    final simdLength = length ~/ 4;
    final lane = Float32x4.splat(value);
    for (var i = 0; i < simdLength; i++) {
      resultSimd[i] = lane;
    }
    for (var i = simdLength * 4; i < length; i++) {
      result[i] = value;
    }
    return result;
  }

  static Float64List elementwise(
    Float64List left,
    Float64List right,
    ElementwiseOp op,
  ) {
    _checkSameLength(left, right);
    final resultSimd = Float64x2List((left.length + 1) ~/ 2);
    final result = resultSimd.buffer.asFloat64List(0, left.length);
    final simdLength = left.length ~/ 2;
    if (simdLength > 0) {
      final leftSimd = left.buffer.asFloat64x2List(
        left.offsetInBytes,
        simdLength,
      );
      final rightSimd = right.buffer.asFloat64x2List(
        right.offsetInBytes,
        simdLength,
      );
      switch (op) {
        case ElementwiseOp.add:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] + rightSimd[i];
          }
        case ElementwiseOp.subtract:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] - rightSimd[i];
          }
        case ElementwiseOp.multiply:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] * rightSimd[i];
          }
        case ElementwiseOp.divide:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] / rightSimd[i];
          }
      }
    }
    for (var i = simdLength * 2; i < left.length; i++) {
      result[i] = switch (op) {
        ElementwiseOp.add => left[i] + right[i],
        ElementwiseOp.subtract => left[i] - right[i],
        ElementwiseOp.multiply => left[i] * right[i],
        ElementwiseOp.divide => left[i] / right[i],
      };
    }
    return result;
  }

  static Float32List elementwise32(
    Float32List left,
    Float32List right,
    ElementwiseOp op,
  ) {
    _checkSameLength32(left, right);
    final resultSimd = Float32x4List((left.length + 3) ~/ 4);
    final result = resultSimd.buffer.asFloat32List(0, left.length);
    final simdLength = left.length ~/ 4;
    if (simdLength > 0) {
      final leftSimd = left.buffer.asFloat32x4List(
        left.offsetInBytes,
        simdLength,
      );
      final rightSimd = right.buffer.asFloat32x4List(
        right.offsetInBytes,
        simdLength,
      );
      switch (op) {
        case ElementwiseOp.add:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] + rightSimd[i];
          }
        case ElementwiseOp.subtract:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] - rightSimd[i];
          }
        case ElementwiseOp.multiply:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] * rightSimd[i];
          }
        case ElementwiseOp.divide:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = leftSimd[i] / rightSimd[i];
          }
      }
    }
    for (var i = simdLength * 4; i < left.length; i++) {
      result[i] = switch (op) {
        ElementwiseOp.add => left[i] + right[i],
        ElementwiseOp.subtract => left[i] - right[i],
        ElementwiseOp.multiply => left[i] * right[i],
        ElementwiseOp.divide => left[i] / right[i],
      };
    }
    return result;
  }

  static Float64List scalar(Float64List data, double value, ElementwiseOp op) {
    if (op == ElementwiseOp.multiply) {
      return scale(data, value);
    }
    final resultSimd = Float64x2List((data.length + 1) ~/ 2);
    final result = resultSimd.buffer.asFloat64List(0, data.length);
    final simdLength = data.length ~/ 2;
    final scalarLane = Float64x2.splat(value);
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat64x2List(
        data.offsetInBytes,
        simdLength,
      );
      switch (op) {
        case ElementwiseOp.add:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] + scalarLane;
          }
        case ElementwiseOp.subtract:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] - scalarLane;
          }
        case ElementwiseOp.multiply:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] * scalarLane;
          }
        case ElementwiseOp.divide:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] / scalarLane;
          }
      }
    }
    for (var i = simdLength * 2; i < data.length; i++) {
      result[i] = switch (op) {
        ElementwiseOp.add => data[i] + value,
        ElementwiseOp.subtract => data[i] - value,
        ElementwiseOp.multiply => data[i] * value,
        ElementwiseOp.divide => data[i] / value,
      };
    }
    return result;
  }

  static Float32List scalar32(
    Float32List data,
    double value,
    ElementwiseOp op,
  ) {
    if (op == ElementwiseOp.multiply) {
      return scale32(data, value);
    }
    final resultSimd = Float32x4List((data.length + 3) ~/ 4);
    final result = resultSimd.buffer.asFloat32List(0, data.length);
    final simdLength = data.length ~/ 4;
    final scalarLane = Float32x4.splat(value);
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat32x4List(
        data.offsetInBytes,
        simdLength,
      );
      switch (op) {
        case ElementwiseOp.add:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] + scalarLane;
          }
        case ElementwiseOp.subtract:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] - scalarLane;
          }
        case ElementwiseOp.multiply:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] * scalarLane;
          }
        case ElementwiseOp.divide:
          for (var i = 0; i < simdLength; i++) {
            resultSimd[i] = dataSimd[i] / scalarLane;
          }
      }
    }
    for (var i = simdLength * 4; i < data.length; i++) {
      result[i] = switch (op) {
        ElementwiseOp.add => data[i] + value,
        ElementwiseOp.subtract => data[i] - value,
        ElementwiseOp.multiply => data[i] * value,
        ElementwiseOp.divide => data[i] / value,
      };
    }
    return result;
  }

  static Float64List scale(Float64List data, double value) {
    final resultSimd = Float64x2List((data.length + 1) ~/ 2);
    final result = resultSimd.buffer.asFloat64List(0, data.length);
    final simdLength = data.length ~/ 2;
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat64x2List(
        data.offsetInBytes,
        simdLength,
      );
      final lane = Float64x2.splat(value);
      var i = 0;
      for (; i + 3 < simdLength; i += 4) {
        resultSimd[i] = dataSimd[i] * lane;
        resultSimd[i + 1] = dataSimd[i + 1] * lane;
        resultSimd[i + 2] = dataSimd[i + 2] * lane;
        resultSimd[i + 3] = dataSimd[i + 3] * lane;
      }
      for (; i < simdLength; i++) {
        resultSimd[i] = dataSimd[i] * lane;
      }
    }
    for (var i = simdLength * 2; i < data.length; i++) {
      result[i] = data[i] * value;
    }
    return result;
  }

  static Float32List scale32(Float32List data, double value) {
    final resultSimd = Float32x4List((data.length + 3) ~/ 4);
    final result = resultSimd.buffer.asFloat32List(0, data.length);
    final simdLength = data.length ~/ 4;
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat32x4List(
        data.offsetInBytes,
        simdLength,
      );
      final lane = Float32x4.splat(value);
      var i = 0;
      for (; i + 3 < simdLength; i += 4) {
        resultSimd[i] = dataSimd[i] * lane;
        resultSimd[i + 1] = dataSimd[i + 1] * lane;
        resultSimd[i + 2] = dataSimd[i + 2] * lane;
        resultSimd[i + 3] = dataSimd[i + 3] * lane;
      }
      for (; i < simdLength; i++) {
        resultSimd[i] = dataSimd[i] * lane;
      }
    }
    for (var i = simdLength * 4; i < data.length; i++) {
      result[i] = data[i] * value;
    }
    return result;
  }

  @pragma('vm:prefer-inline')
  static double dot(
    Float64List left,
    int leftOffset,
    Float64List right,
    int rightOffset,
    int length,
  ) {
    final leftByteOffset =
        left.offsetInBytes + leftOffset * Float64List.bytesPerElement;
    final rightByteOffset =
        right.offsetInBytes + rightOffset * Float64List.bytesPerElement;
    if (leftByteOffset % Float64x2List.bytesPerElement == 0 &&
        rightByteOffset % Float64x2List.bytesPerElement == 0) {
      final simdLength = length ~/ 2;
      if (simdLength > 0) {
        final leftSimd = left.buffer.asFloat64x2List(
          leftByteOffset,
          simdLength,
        );
        final rightSimd = right.buffer.asFloat64x2List(
          rightByteOffset,
          simdLength,
        );
        var lanes0 = Float64x2.zero();
        var lanes1 = Float64x2.zero();
        var lanes2 = Float64x2.zero();
        var lanes3 = Float64x2.zero();
        var i = 0;
        for (; i + 3 < simdLength; i += 4) {
          lanes0 += leftSimd[i] * rightSimd[i];
          lanes1 += leftSimd[i + 1] * rightSimd[i + 1];
          lanes2 += leftSimd[i + 2] * rightSimd[i + 2];
          lanes3 += leftSimd[i + 3] * rightSimd[i + 3];
        }
        var lanes = lanes0 + lanes1 + lanes2 + lanes3;
        for (; i < simdLength; i++) {
          lanes += leftSimd[i] * rightSimd[i];
        }
        var result = lanes.x + lanes.y;
        for (var i = simdLength * 2; i < length; i++) {
          result += left[leftOffset + i] * right[rightOffset + i];
        }
        return result;
      }
    }
    var lanes = Float64x2.zero();
    var i = 0;
    for (; i + 1 < length; i += 2) {
      lanes +=
          Float64x2(left[leftOffset + i], left[leftOffset + i + 1]) *
          Float64x2(right[rightOffset + i], right[rightOffset + i + 1]);
    }
    var result = lanes.x + lanes.y;
    for (; i < length; i++) {
      result += left[leftOffset + i] * right[rightOffset + i];
    }
    return result;
  }

  @pragma('vm:prefer-inline')
  static double dot32(
    Float32List left,
    int leftOffset,
    Float32List right,
    int rightOffset,
    int length,
  ) {
    final leftByteOffset =
        left.offsetInBytes + leftOffset * Float32List.bytesPerElement;
    final rightByteOffset =
        right.offsetInBytes + rightOffset * Float32List.bytesPerElement;
    if (leftByteOffset % Float32x4List.bytesPerElement == 0 &&
        rightByteOffset % Float32x4List.bytesPerElement == 0) {
      final simdLength = length ~/ 4;
      if (simdLength > 0) {
        final leftSimd = left.buffer.asFloat32x4List(
          leftByteOffset,
          simdLength,
        );
        final rightSimd = right.buffer.asFloat32x4List(
          rightByteOffset,
          simdLength,
        );
        var lanes0 = Float32x4.zero();
        var lanes1 = Float32x4.zero();
        var lanes2 = Float32x4.zero();
        var lanes3 = Float32x4.zero();
        var i = 0;
        for (; i + 3 < simdLength; i += 4) {
          lanes0 += leftSimd[i] * rightSimd[i];
          lanes1 += leftSimd[i + 1] * rightSimd[i + 1];
          lanes2 += leftSimd[i + 2] * rightSimd[i + 2];
          lanes3 += leftSimd[i + 3] * rightSimd[i + 3];
        }
        var lanes = lanes0 + lanes1 + lanes2 + lanes3;
        for (; i < simdLength; i++) {
          lanes += leftSimd[i] * rightSimd[i];
        }
        var result = lanes.x + lanes.y + lanes.z + lanes.w;
        for (var i = simdLength * 4; i < length; i++) {
          result += left[leftOffset + i] * right[rightOffset + i];
        }
        return result;
      }
    }
    var lanes = Float32x4.zero();
    var i = 0;
    for (; i + 3 < length; i += 4) {
      final a = Float32x4(
        left[leftOffset + i],
        left[leftOffset + i + 1],
        left[leftOffset + i + 2],
        left[leftOffset + i + 3],
      );
      final b = Float32x4(
        right[rightOffset + i],
        right[rightOffset + i + 1],
        right[rightOffset + i + 2],
        right[rightOffset + i + 3],
      );
      lanes += a * b;
    }
    var result = lanes.x + lanes.y + lanes.z + lanes.w;
    for (; i < length; i++) {
      result += left[leftOffset + i] * right[rightOffset + i];
    }
    return result;
  }

  static Float64List mv(
    Float64List matrix,
    int rows,
    int columns,
    Float64List vector,
  ) {
    if (columns != vector.length) {
      throw ArgumentError(
        'Matrix64/vector shape mismatch: ($rows, $columns) x ${vector.length}.',
      );
    }
    final result = Float64List(rows);
    final matrixByteOffset = matrix.offsetInBytes;
    final vectorByteOffset = vector.offsetInBytes;
    if (columns % 2 == 0 &&
        matrixByteOffset % Float64x2List.bytesPerElement == 0 &&
        vectorByteOffset % Float64x2List.bytesPerElement == 0) {
      final simdColumns = columns ~/ 2;
      final matrixSimd = matrix.buffer.asFloat64x2List(
        matrixByteOffset,
        rows * simdColumns,
      );
      final vectorSimd = vector.buffer.asFloat64x2List(
        vectorByteOffset,
        simdColumns,
      );
      for (var row = 0; row < rows; row++) {
        final offset = row * simdColumns;
        var lanes0 = Float64x2.zero();
        var lanes1 = Float64x2.zero();
        var lanes2 = Float64x2.zero();
        var lanes3 = Float64x2.zero();
        var column = 0;
        for (; column + 3 < simdColumns; column += 4) {
          lanes0 += matrixSimd[offset + column] * vectorSimd[column];
          lanes1 += matrixSimd[offset + column + 1] * vectorSimd[column + 1];
          lanes2 += matrixSimd[offset + column + 2] * vectorSimd[column + 2];
          lanes3 += matrixSimd[offset + column + 3] * vectorSimd[column + 3];
        }
        var lanes = lanes0 + lanes1 + lanes2 + lanes3;
        for (; column < simdColumns; column++) {
          lanes += matrixSimd[offset + column] * vectorSimd[column];
        }
        result[row] = lanes.x + lanes.y;
      }
      return result;
    }
    for (var row = 0; row < rows; row++) {
      result[row] = dot(matrix, row * columns, vector, 0, columns);
    }
    return result;
  }

  static Float32List mv32(
    Float32List matrix,
    int rows,
    int columns,
    Float32List vector,
  ) {
    if (columns != vector.length) {
      throw ArgumentError(
        'Matrix64/vector shape mismatch: ($rows, $columns) x ${vector.length}.',
      );
    }
    final result = Float32List(rows);
    final matrixByteOffset = matrix.offsetInBytes;
    final vectorByteOffset = vector.offsetInBytes;
    if (columns % 4 == 0 &&
        matrixByteOffset % Float32x4List.bytesPerElement == 0 &&
        vectorByteOffset % Float32x4List.bytesPerElement == 0) {
      final simdColumns = columns ~/ 4;
      final matrixSimd = matrix.buffer.asFloat32x4List(
        matrixByteOffset,
        rows * simdColumns,
      );
      final vectorSimd = vector.buffer.asFloat32x4List(
        vectorByteOffset,
        simdColumns,
      );
      for (var row = 0; row < rows; row++) {
        final offset = row * simdColumns;
        var lanes0 = Float32x4.zero();
        var lanes1 = Float32x4.zero();
        var lanes2 = Float32x4.zero();
        var lanes3 = Float32x4.zero();
        var column = 0;
        for (; column + 3 < simdColumns; column += 4) {
          lanes0 += matrixSimd[offset + column] * vectorSimd[column];
          lanes1 += matrixSimd[offset + column + 1] * vectorSimd[column + 1];
          lanes2 += matrixSimd[offset + column + 2] * vectorSimd[column + 2];
          lanes3 += matrixSimd[offset + column + 3] * vectorSimd[column + 3];
        }
        var lanes = lanes0 + lanes1 + lanes2 + lanes3;
        for (; column < simdColumns; column++) {
          lanes += matrixSimd[offset + column] * vectorSimd[column];
        }
        result[row] = lanes.x + lanes.y + lanes.z + lanes.w;
      }
      return result;
    }
    for (var row = 0; row < rows; row++) {
      result[row] = dot32(matrix, row * columns, vector, 0, columns);
    }
    return result;
  }

  static Float64List transpose(Float64List data, int rows, int columns) {
    if (data.length <= 16384 || data.length >= 1000000) {
      return _transposeSimple(data, rows, columns);
    }
    final result = Float64List(data.length);
    const block = 16;
    for (var rowBlock = 0; rowBlock < rows; rowBlock += block) {
      final rowLimit = _min(rowBlock + block, rows);
      for (var columnBlock = 0; columnBlock < columns; columnBlock += block) {
        final columnLimit = _min(columnBlock + block, columns);
        for (var row = rowBlock; row < rowLimit; row++) {
          final sourceOffset = row * columns;
          for (var column = columnBlock; column < columnLimit; column++) {
            result[column * rows + row] = data[sourceOffset + column];
          }
        }
      }
    }
    return result;
  }

  static Float32List transpose32(Float32List data, int rows, int columns) {
    if (data.length <= 16384 || data.length >= 1000000) {
      return _transposeSimple32(data, rows, columns);
    }
    final result = Float32List(data.length);
    const block = 16;
    for (var rowBlock = 0; rowBlock < rows; rowBlock += block) {
      final rowLimit = _min(rowBlock + block, rows);
      for (var columnBlock = 0; columnBlock < columns; columnBlock += block) {
        final columnLimit = _min(columnBlock + block, columns);
        for (var row = rowBlock; row < rowLimit; row++) {
          final sourceOffset = row * columns;
          for (var column = columnBlock; column < columnLimit; column++) {
            result[column * rows + row] = data[sourceOffset + column];
          }
        }
      }
    }
    return result;
  }

  static Float64List _transposeSimple(Float64List data, int rows, int columns) {
    final result = Float64List(data.length);
    for (var row = 0; row < rows; row++) {
      final sourceOffset = row * columns;
      for (var column = 0; column < columns; column++) {
        result[column * rows + row] = data[sourceOffset + column];
      }
    }
    return result;
  }

  static Float32List _transposeSimple32(
    Float32List data,
    int rows,
    int columns,
  ) {
    final result = Float32List(data.length);
    for (var row = 0; row < rows; row++) {
      final sourceOffset = row * columns;
      for (var column = 0; column < columns; column++) {
        result[column * rows + row] = data[sourceOffset + column];
      }
    }
    return result;
  }

  static double sum(Float64List data) {
    var lanes = Float64x2.zero();
    final simdLength = data.length ~/ 2;
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat64x2List(
        data.offsetInBytes,
        simdLength,
      );
      var lanes0 = Float64x2.zero();
      var lanes1 = Float64x2.zero();
      var lanes2 = Float64x2.zero();
      var lanes3 = Float64x2.zero();
      var i = 0;
      for (; i + 3 < simdLength; i += 4) {
        lanes0 += dataSimd[i];
        lanes1 += dataSimd[i + 1];
        lanes2 += dataSimd[i + 2];
        lanes3 += dataSimd[i + 3];
      }
      lanes = lanes0 + lanes1 + lanes2 + lanes3;
      for (; i < simdLength; i++) {
        lanes += dataSimd[i];
      }
    }
    var result = lanes.x + lanes.y;
    for (var i = simdLength * 2; i < data.length; i++) {
      result += data[i];
    }
    return result;
  }

  static double sum32(Float32List data) {
    var lanes = Float32x4.zero();
    final simdLength = data.length ~/ 4;
    if (simdLength > 0) {
      final dataSimd = data.buffer.asFloat32x4List(
        data.offsetInBytes,
        simdLength,
      );
      for (var i = 0; i < simdLength; i++) {
        lanes += dataSimd[i];
      }
    }
    var result = lanes.x + lanes.y + lanes.z + lanes.w;
    for (var i = simdLength * 4; i < data.length; i++) {
      result += data[i];
    }
    return result;
  }

  static void _checkSameLength(Float64List left, Float64List right) {
    if (left.length != right.length) {
      throw ArgumentError.value(
        right.length,
        'right.length',
        'Expected ${left.length}.',
      );
    }
  }

  static void _checkSameLength32(Float32List left, Float32List right) {
    if (left.length != right.length) {
      throw ArgumentError.value(
        right.length,
        'right.length',
        'Expected ${left.length}.',
      );
    }
  }

  static int _min(int a, int b) => a < b ? a : b;
}
