import '../lib/matrices.dart';

void main() {
  final a = Matrix64([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  final b = Matrix64([
    [7, 8],
    [9, 10],
    [11, 12],
  ]);

  print(a);
  print(a * b);

  final x = mat([
    [3, 2],
    [1, 2],
  ]);
  final y = mat([
    [5],
    [5],
  ]);

  print(x.solve(y));
  print(eye(3));
  print(diag([1, 2, 3]));
  print(arange(0, 9, columns: 3));
  print(linspace(0, 1, 5));

  final square = SquareMatrix.fromList([
    [4, 7],
    [2, 6],
  ]);
  print(square.determinant);
  print(square.inverse);

  final lu = square.lu();
  print(lu.lower * lu.upper);

  final spd = mat([
    [4, 1],
    [1, 3],
  ]);
  final rhs = vec([1, 2]);
  print(spd.cholesky().lower);
  print(spd.conjugateGradient(rhs).solution);
  print(spd.powerIteration().eigenvalue);
}
