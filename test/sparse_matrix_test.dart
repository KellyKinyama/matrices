import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('constructs CSR sparse matrix from dense rows', () {
    final sparse = SparseMatrix.fromRows([
      [1, 0, 2],
      [0, 0, 3],
      [4, 0, 0],
    ]);

    expect(sparse.shape, (3, 3));
    expect(sparse.nnz, 4);
    expect(sparse.density, closeTo(4 / 9, 1e-12));
    expect(sparse(0, 2), 2);
    expect(sparse(1, 1), 0);
    expect(sparse.row(0), [1, 0, 2]);
    expect(sparse.column(2), [2, 3, 0]);
  });

  test('converts between dense and sparse with tolerance', () {
    final dense = Matrix64([
      [1, 1e-12],
      [0, 2],
    ]);

    final sparse = dense.toSparse(tolerance: 1e-9);

    expect(sparse.nnz, 2);
    expect(sparse.toDense().toRows(), [
      [1, 0],
      [0, 2],
    ]);
  });

  test('builds from triplets and merges duplicates', () {
    final sparse = SparseMatrix.fromTriplets(2, 3, [
      SparseEntry(0, 1, 2),
      SparseEntry(0, 1, 3),
      SparseEntry(1, 2, 4),
    ]);

    expect(sparse.nnz, 2);
    expect(sparse.toDense().toRows(), [
      [0, 5, 0],
      [0, 0, 4],
    ]);
  });

  test('multiplies sparse matrix by vector and dense matrix', () {
    final sparse = SparseMatrix.fromRows([
      [1, 0, 2],
      [0, 3, 0],
    ]);
    final vectorResult = sparse.mv(vec([1, 2, 3]));
    final matrixResult = sparse.matmul(
      Matrix64([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
    );

    expect(vectorResult, [7, 6]);
    expect(matrixResult.toRows(), [
      [11, 14],
      [9, 12],
    ]);
  });

  test('transposes and serializes sparse matrices', () {
    final sparse = SparseMatrix.fromRows([
      [0, 1, 0],
      [2, 0, 3],
    ]);

    final transposed = sparse.transpose();
    final decoded = SparseMatrix.fromJson(sparse.toJson());

    expect(transposed.toDense().toRows(), [
      [0, 2],
      [1, 0],
      [0, 3],
    ]);
    expect(decoded.toDense().toRows(), sparse.toDense().toRows());
  });
}
