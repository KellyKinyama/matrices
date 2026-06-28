import 'dart:typed_data';

import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('Matrix and Vector are script-friendly aliases for 64-bit types', () {
    final aliasMatrix = Matrix([
      [1, 2],
      [3, 4],
    ]);
    final aliasVector = Vector([1, 2]);
    final fullWordMatrix = matrix([
      [1, 2],
      [3, 4],
    ]);
    final fullWordMatrix32 = matrix32([
      [1, 2],
      [3, 4],
    ]);
    final fullWordVector = vector([1, 2]);
    final fullWordVector32 = vector32([1, 2]);
    final shortVector = vec([1, 2]);
    final shortVector32 = vec32([1, 2]);

    expect(aliasMatrix, isA<Matrix64>());
    expect(aliasVector, isA<Vector64>());
    expect(fullWordMatrix, isA<Matrix64>());
    expect(fullWordMatrix32, isA<Matrix32>());
    expect(fullWordVector, isA<Vector64>());
    expect(fullWordVector32, isA<Vector32>());
    expect(shortVector, isA<Vector64>());
    expect(shortVector32, isA<Vector32>());
    expect(vectorZeros(2), [0, 0]);
    expect(vectorOnes(2), [1, 1]);
    expect(vectorFull(2, 3), [3, 3]);
    expect(randomVector(2, seed: 1), hasLength(2));
    expect(vectorRange(0, 4, step: 2), [0, 2]);
    expect(vectorLinspace(0, 1, 3), [0, 0.5, 1]);
    expect(vecZeros(2), [0, 0]);
    expect(vecOnes(2), [1, 1]);
    expect(vecFull(2, 3), [3, 3]);
    expect(vecRange(0, 4, step: 2), [0, 2]);
    expect(vecLinspace(0, 1, 3), [0, 0.5, 1]);
    expect(vecZeros32(2), [0, 0]);
    expect(vecOnes32(2), [1, 1]);
    expect(vecFull32(2, 3), [3, 3]);
    expect((aliasMatrix * aliasVector), isA<Vector64>());
    expect((aliasMatrix * aliasVector), [5, 11]);
    expect((aliasMatrix * shortVector), [5, 11]);
  });

  test('Matrix64 exposes data-science construction APIs', () {
    final columns = Matrix64.fromColumns([
      [1, 2],
      [3, 4],
    ]);
    expect(columns.toRows(), [
      [1, 3],
      [2, 4],
    ]);

    final bytes = Float64List.fromList([1, 2, 3, 4]).buffer.asByteData();
    final fromBytes = Matrix64.fromByteData(bytes, 2, 2);
    expect(fromBytes(1, 1), 4);

    final spd = Matrix64.randomSPD(3, seed: 3);
    expect(spd.cholesky().lower.rows, 3);

    final scalar = Matrix64.scalar(5, 2);
    expect(scalar.toRows(), [
      [5, 0],
      [0, 5],
    ]);
  });

  test('Matrix64 supports statistics mapping filtering sorting and json', () {
    final a = Matrix64([
      [1, 2, 2],
      [3, 4, 4],
      [1, 2, 2],
    ]);

    expect(a.meanByAxis(Axis.columns), [5 / 3, 8 / 3, 8 / 3]);
    expect(a.meanByAxis(Axis.rows), [5 / 3, 11 / 3, 5 / 3]);
    expect(a.variance(Axis.rows)[0], closeTo(2 / 9, 1e-12));
    expect(a.deviation(Axis.rows)[0], closeTo(0.47140452079, 1e-10));
    expect(a.prod(), 768);
    expect(a.pow(2)(0, 1), 4);
    expect(a.sqrt()(1, 0), closeTo(1.73205080757, 1e-10));

    expect(a.sample(rowIndices: [1], columnIndices: [0, 2]).toRows(), [
      [3, 4],
    ]);
    expect(a.uniqueRows().rows, 2);
    expect(a.filterRows((row, index) => row[0] > 1).rows, 1);
    expect(a.filterColumns((column, index) => index != 1).columns, 2);
    expect(a.mapRows((row) => row + 1)(0, 0), 2);
    expect(a.mapColumns((column) => column * 2)(1, 1), 8);
    expect(a.sort((row) => row.sum, direction: SortDirection.desc)(0, 0), 3);

    final encoded = a.toJson();
    expect(Matrix64.fromJson(encoded).toRows(), a.toRows());
  });

  test('Matrix32 mirrors Matrix64 high-level APIs', () {
    final a = Matrix32([
      [4, 1],
      [1, 3],
    ]);

    expect(a.transpose.toRows(), [
      [4, 1],
      [1, 3],
    ]);
    expect(Matrix32.ones(1, 2).toRows(), [
      [1, 1],
    ]);
    expect(ones32(1, 2).toRows(), [
      [1, 1],
    ]);
    expect(eye32(2).toRows(), [
      [1, 0],
      [0, 1],
    ]);
    expect(diag32([2, 3]).toRows(), [
      [2, 0],
      [0, 3],
    ]);
    expect(arange32(0, 4, columns: 2).toRows(), [
      [0, 1],
      [2, 3],
    ]);
    expect(linspace32(0, 1, 3).toRows(), [
      [0],
      [0.5],
      [1],
    ]);
    expect(a.reshape(1, 4).shape, (1, 4));
    expect(a.slice(rowStart: 0, rowEnd: 1).shape, (1, 2));
    expect(a.meanByAxis(Axis.columns), [2.5, 2.0]);
    expect(a.variance(Axis.rows)[0], closeTo(2.25, 1e-6));
    expect(a.pow(2)(1, 1), closeTo(9, 1e-6));
    expect(a.sqrt()(0, 0), closeTo(2, 1e-6));
    expect(a.hstack(a).shape, (2, 4));
    expect(a.vstack(a).shape, (4, 2));
    expect(a.inverse(0, 0), closeTo(3 / 11, 1e-5));
    expect(a.cholesky().lower.rows, 2);
    expect(a.conjugateGradient(vec([1, 2])).converged, isTrue);
    expect(a.toSparse().toDense().toRows(), a.toFloat64().toRows());

    final encoded = a.toJson();
    expect(Matrix32.fromJson(encoded).toRows(), a.toRows());
  });

  test('Vector64 exposes data-science helpers', () {
    final x = vec([1, 2, 2, 4]);
    final y = vec([1, 1, 3, 5]);

    expect(x.median(), 2);
    expect(x.prod(), 16);
    expect(x.maxValue(), 4);
    expect(x.minValue(), 1);
    expect(x.distanceTo(y, Distance.manhattan), 3);
    expect(x.distanceTo(y, Distance.hamming), 3);
    expect(x.cosine(y), closeTo(29 / 30, 1e-10));
    expect(x.normalize().norm(), closeTo(1, 1e-10));
    expect(x.rescale(), [0, 1 / 3, 1 / 3, 1]);
    expect(x.unique(), [1, 2, 4]);
    expect(x.sample([3, 0]), [4, 1]);
    expect(x.subvector(1, 3), [2, 2]);
    expect(x.filterElements((value, index) => value > 1), [2, 2, 4]);
    expect(x.pow(2), [1, 4, 4, 16]);
    expect(Vector64.fromJson(x.toJson()), x);
  });
}
