import 'dart:math' as math;

import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('matmul backward matches finite differences', () {
    final a = Variable.fromRows([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ], requiresGrad: true);
    final b = Variable.fromRows([
      [0.5, -1.0],
      [2.0, 1.5],
      [-0.5, 0.25],
    ], requiresGrad: true);

    double loss(Matrix64 av, Matrix64 bv) => av.matmul(bv).sum;

    (a.matmul(b)).sum().backward();

    _checkGradient(a, b, loss);
  });

  test('nonlinear scalar graph matches finite differences', () {
    final x = Variable.fromRows([
      [0.3, -0.7],
      [1.2, 0.4],
    ], requiresGrad: true);

    Variable forward(Variable v) =>
        v.sigmoid().matmul(v.transpose).relu().sum();

    forward(x).backward();

    final analytic = x.grad!;
    final numeric = _numericalGradient(
      x.value,
      (m) => forward(Variable(m)).item,
    );
    _expectClose(analytic, numeric, 1e-5);
  });

  test('linear regression converges via gradient descent', () {
    final rng = math.Random(7);
    const samples = 40;
    const features = 3;

    final trueWeights = Variable.fromRows([
      [2.0],
      [-3.0],
      [0.5],
    ]);

    final xRows = List<List<double>>.generate(
      samples,
      (_) => List<double>.generate(features, (_) => rng.nextDouble() * 2 - 1),
    );
    final x = Variable.fromRows(xRows);
    final y = x.matmul(trueWeights);

    final w = Variable.zeros(features, 1, requiresGrad: true);
    const learningRate = 0.5;
    var lastLoss = double.infinity;

    for (var step = 0; step < 400; step++) {
      w.zeroGrad();
      final prediction = x.matmul(w);
      final error = prediction - y;
      final loss = (error * error).mean();
      loss.backward();

      final gradient = w.grad!;
      Variable.noGrad(() {
        w.value = w.value - gradient.scale(learningRate);
      });
      lastLoss = loss.item;
    }

    expect(lastLoss, lessThan(1e-8));
    for (var i = 0; i < features; i++) {
      expect(w.value(i, 0), closeTo(trueWeights.value(i, 0), 1e-3));
    }
  });

  test('noGrad suppresses graph construction', () {
    final x = Variable.scalar(2.0, requiresGrad: true);
    final y = Variable.noGrad(() => x.scale(3));
    expect(y.requiresGrad, isFalse);
  });

  test('addRowVector backward reduces bias gradient over rows', () {
    final x = Variable.fromRows([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ], requiresGrad: true);
    final bias = Variable.fromRows([
      [0.5, -1.0, 2.0],
    ], requiresGrad: true);

    x.addRowVector(bias).sum().backward();

    // d/dx sum(x + bias) = ones like x.
    _expectClose(x.grad!, Matrix64.full(2, 3, 1.0), 1e-9);
    // d/dbias = column sums of the ones gradient = number of rows.
    _expectClose(bias.grad!, Matrix64.full(1, 3, 2.0), 1e-9);
  });

  test('backward frees intermediate buffers by default', () {
    final w = Variable.fromRows([
      [1.0],
      [2.0],
    ], requiresGrad: true);
    final x = Variable.fromRows([
      [3.0, 4.0],
    ]);

    final hidden = x.matmul(w);
    final loss = hidden.sum();
    loss.backward();

    // Intermediate value + grad released; parameter grad retained.
    expect(hidden.value.count, 0);
    expect(hidden.grad, isNull);
    expect(w.grad, isNotNull);
  });

  test('retainGraph keeps intermediates alive', () {
    final w = Variable.fromRows([
      [1.0],
      [2.0],
    ], requiresGrad: true);
    final x = Variable.fromRows([
      [3.0, 4.0],
    ]);

    final hidden = x.matmul(w);
    hidden.sum().backward(retainGraph: true);

    expect(hidden.value.count, greaterThan(0));
  });
}

void _checkGradient(
  Variable a,
  Variable b,
  double Function(Matrix64, Matrix64) loss,
) {
  final numericA = _numericalGradient(a.value, (m) => loss(m, b.value));
  final numericB = _numericalGradient(b.value, (m) => loss(a.value, m));
  _expectClose(a.grad!, numericA, 1e-5);
  _expectClose(b.grad!, numericB, 1e-5);
}

Matrix64 _numericalGradient(Matrix64 input, double Function(Matrix64) loss) {
  const epsilon = 1e-6;
  final base = input.values;
  final grad = List<double>.filled(base.length, 0);
  for (var i = 0; i < base.length; i++) {
    final plus = base.sublist(0);
    final minus = base.sublist(0);
    plus[i] += epsilon;
    minus[i] -= epsilon;
    final lossPlus = loss(Matrix64.fromFlat(plus, input.rows, input.columns));
    final lossMinus = loss(Matrix64.fromFlat(minus, input.rows, input.columns));
    grad[i] = (lossPlus - lossMinus) / (2 * epsilon);
  }
  return Matrix64.fromFlat(grad, input.rows, input.columns);
}

void _expectClose(Matrix64 actual, Matrix64 expected, double tolerance) {
  expect(actual.rows, expected.rows);
  expect(actual.columns, expected.columns);
  for (var r = 0; r < actual.rows; r++) {
    for (var c = 0; c < actual.columns; c++) {
      expect(actual(r, c), closeTo(expected(r, c), tolerance));
    }
  }
}
