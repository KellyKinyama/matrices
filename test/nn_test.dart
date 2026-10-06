import 'package:matrices/matrices.dart';
import 'package:test/test.dart';

void main() {
  test('Linear forward applies weight and bias', () {
    final layer = Linear(2, 3, seed: 1);
    final input = Variable.fromRows([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);

    final output = layer(input);
    expect(output.rows, 2);
    expect(output.columns, 3);
    expect(layer.parameters, hasLength(2));
  });

  test('MLP learns a nonlinear XOR-like mapping', () {
    final model = Sequential([
      Linear(2, 8, seed: 3),
      Tanh(),
      Linear(8, 1, seed: 4),
    ]);
    final optimizer = SGD(model.parameters, learningRate: 0.1);

    final inputs = Variable.fromRows([
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0],
    ]);
    final targets = Variable.fromRows([
      [0.0],
      [1.0],
      [1.0],
      [0.0],
    ]);

    var lastLoss = double.infinity;
    for (var epoch = 0; epoch < 2000; epoch++) {
      optimizer.zeroGrad();
      final prediction = model(inputs);
      final error = prediction - targets;
      final loss = (error * error).mean();
      loss.backward();
      optimizer.step();
      lastLoss = loss.item;
    }

    expect(lastLoss, lessThan(1e-3));

    final prediction = Variable.noGrad(() => model(inputs));
    expect(prediction.value(0, 0), closeTo(0, 0.1));
    expect(prediction.value(1, 0), closeTo(1, 0.1));
    expect(prediction.value(2, 0), closeTo(1, 0.1));
    expect(prediction.value(3, 0), closeTo(0, 0.1));
  });

  test('SGD with momentum minimizes a quadratic', () {
    final w = Variable.fromRows([
      [5.0, -3.0],
    ], requiresGrad: true);
    final optimizer = SGD([w], learningRate: 0.01, momentum: 0.9);

    for (var step = 0; step < 300; step++) {
      optimizer.zeroGrad();
      // Minimize sum(w^2); minimum at the origin.
      (w * w).sum().backward();
      optimizer.step();
    }

    expect(w.value(0, 0), closeTo(0, 1e-3));
    expect(w.value(0, 1), closeTo(0, 1e-3));
  });

  test('Adam minimizes a quadratic toward the origin', () {
    final w = Variable.fromRows([
      [5.0, -3.0],
    ], requiresGrad: true);
    final optimizer = Adam([w], learningRate: 0.1);

    for (var step = 0; step < 400; step++) {
      optimizer.zeroGrad();
      (w * w).sum().backward();
      optimizer.step();
    }

    expect(w.value(0, 0), closeTo(0, 1e-3));
    expect(w.value(0, 1), closeTo(0, 1e-3));
  });
}
