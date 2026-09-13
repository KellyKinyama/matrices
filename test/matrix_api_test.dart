import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('constructors produce expected shapes and values', () {
    final Matrix64 explicit64 = Matrix64([
      [1, 2],
      [3, 4],
    ]);

    expect(explicit64, isA<Matrix64>());
    expect(explicit64.shape, (2, 2));
    expect(zeros(2, 3).shape, (2, 3));
    expect(ones(2, 2)(1, 1), 1);
    expect(full(2, 2, 9)(0, 1), 9);
    expect(eye(3)(2, 2), 1);
    expect(diag([1, 2, 3])(1, 1), 2);
    expect(arange(0, 6, columns: 3).shape, (2, 3));
    expect(linspace(0, 1, 5)(4, 0), 1);
  });

  test('row mutation invalidates transpose cache used by matmul', () {
    final a = mat([
      [1, 0],
      [0, 1],
    ]);
    final b = mat([
      [2, 3],
      [4, 5],
    ]);

    expect((a * b)(0, 0), 2);
    b.set(0, 0, 10);
    expect((a * b)(0, 0), 10);
  });

  test('slice reshape stack and flatten work together', () {
    final a = arange(0, 9, columns: 3);

    expect(a.slice(rowStart: 1, rowEnd: 3, columnStart: 1)(0, 0), 4);
    expect(a.reshape(1, 9).shape, (1, 9));
    expect(a.vstack(a).shape, (6, 3));
    expect(a.hstack(a).shape, (3, 6));
    expect(a.flatten(), hasLength(9));
  });

  test('vector helpers cover arithmetic and norms', () {
    final x = vec([1, 2, 3]);
    final y = vec([4, 5, 6]);

    expect(x.dot(y), 32);
    expect((x + y)[2], 9);
    expect((x * 2)[1], 4);
    expect(x.norm(), closeTo(3.74165738677, 1e-10));
    expect(vecLinspace(0, 1, 3), [0, 0.5, 1]);
  });

  test('matrix and vector operators support script-style linear algebra', () {
    final a = mat([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    final row = vec([10, 20, 30]);
    final column = vec([2, 3]);
    final x = vec([1, 1, 1]);
    final left = vec([1, 2]);

    expect((a + row).toRows(), [
      [11, 22, 33],
      [14, 25, 36],
    ]);
    expect((row + a).toRows(), [
      [11, 22, 33],
      [14, 25, 36],
    ]);
    expect((a - column).toRows(), [
      [-1, 0, 1],
      [1, 2, 3],
    ]);
    expect((column - a).toRows(), [
      [1, 0, -1],
      [-1, -2, -3],
    ]);
    expect(a * x, [6, 15]);
    expect(left * a, [9, 12, 15]);
  });

  test('shape and numerical errors are explicit', () {
    expect(
      () =>
          mat([
            [1, 2],
          ]) *
          mat([
            [1, 2],
          ]),
      throwsArgumentError,
    );
    expect(
      () => mat([
        [1, 2],
        [2, 4],
      ]).inverse,
      throwsStateError,
    );
    expect(
      () => mat([
        [1, 2],
        [3, 4],
      ]).cholesky(),
      throwsStateError,
    );
    expect(
      () =>
          mat32([
            [1, 2],
          ]) +
          mat32([
            [1],
            [2],
          ]),
      throwsArgumentError,
    );
  });

  test('typed arithmetic methods preserve static types', () {
    final Matrix64 a = mat([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    final Matrix64 scaled = a.scale(2);
    expect(scaled.transpose.toRows(), [
      [2, 8],
      [4, 10],
      [6, 12],
    ]);

    final Vector64 x = vec([1, 2, 3]);
    final Vector64 doubled = x.scale(2);
    expect(doubled.toList(), [2, 4, 6]);

    final Matrix32 a32 = mat32([
      [1, 2],
      [3, 4],
    ]);
    expect(a32.scale(3).transpose.toRows(), [
      [3, 9],
      [6, 12],
    ]);
    expect(vec32([1, 2]).scale(4).toList(), [4, 8]);
  });

  test('explicit axis broadcasting removes square-matrix ambiguity', () {
    final a = mat([
      [1, 2],
      [3, 4],
    ]);
    final v = vec([10, 20]);

    expect(a.addVector(v, Axis.columns).toRows(), [
      [11, 22],
      [13, 24],
    ]);
    expect(a.addVector(v, Axis.rows).toRows(), [
      [11, 12],
      [23, 24],
    ]);
    expect(a.subtractVector(v, Axis.columns).toRows(), [
      [-9, -18],
      [-7, -16],
    ]);

    final a32 = mat32([
      [1, 2],
      [3, 4],
    ]);
    expect(a32.multiplyVector(vec32([2, 3]), Axis.rows).toRows(), [
      [2, 4],
      [9, 12],
    ]);

    expect(
      () => a.addVector(vec([1, 2, 3]), Axis.columns),
      throwsArgumentError,
    );
  });
}
