import 'dart:math' as math;

import 'autodiff.dart';
import 'matrix.dart';

/// Base class for differentiable, parameterized layers built on
/// [Variable]. Instances are callable: `layer(input)`.
abstract class Module {
  /// Trainable parameters exposed to an optimizer.
  List<Variable> get parameters;

  /// Forward pass.
  Variable call(Variable input);

  /// Clears gradients on all parameters.
  void zeroGrad() {
    for (final parameter in parameters) {
      parameter.zeroGrad();
    }
  }
}

/// Fully connected layer computing `input @ weight + bias`.
///
/// `weight` is `(inFeatures, outFeatures)` initialized with Xavier
/// uniform sampling; `bias` is `(1, outFeatures)` initialized to zero
/// and broadcast across rows.
class Linear extends Module {
  final Variable weight;
  final Variable bias;

  Linear(int inFeatures, int outFeatures, {int? seed})
    : weight = _initWeight(inFeatures, outFeatures, seed),
      bias = Variable.zeros(1, outFeatures, requiresGrad: true);

  @override
  Variable call(Variable input) => input.matmul(weight).addRowVector(bias);

  @override
  List<Variable> get parameters => [weight, bias];

  static Variable _initWeight(int inFeatures, int outFeatures, int? seed) {
    final limit = math.sqrt(6.0 / (inFeatures + outFeatures));
    final base = Matrix64.random(inFeatures, outFeatures, seed: seed);
    return Variable(
      base.mapValues((value) => (value * 2 - 1) * limit),
      requiresGrad: true,
    );
  }
}

class ReLU extends Module {
  @override
  Variable call(Variable input) => input.relu();

  @override
  List<Variable> get parameters => const [];
}

class Tanh extends Module {
  @override
  Variable call(Variable input) => input.tanh();

  @override
  List<Variable> get parameters => const [];
}

class Sigmoid extends Module {
  @override
  Variable call(Variable input) => input.sigmoid();

  @override
  List<Variable> get parameters => const [];
}

/// Runs its [layers] in order, threading the output of each into the
/// next.
class Sequential extends Module {
  final List<Module> layers;

  Sequential(this.layers);

  @override
  Variable call(Variable input) {
    var output = input;
    for (final layer in layers) {
      output = layer(output);
    }
    return output;
  }

  @override
  List<Variable> get parameters => [
    for (final layer in layers) ...layer.parameters,
  ];
}

/// Base class for gradient-descent optimizers over a parameter list.
abstract class Optimizer {
  List<Variable> get parameters;

  /// Applies one update step from the current gradients.
  void step();

  /// Clears gradients on all parameters.
  void zeroGrad() {
    for (final parameter in parameters) {
      parameter.zeroGrad();
    }
  }
}

/// Vanilla stochastic gradient descent with optional [momentum].
/// [step] updates parameters in a [Variable.noGrad] scope so the update
/// is not recorded on the graph.
class SGD extends Optimizer {
  @override
  final List<Variable> parameters;
  final double learningRate;
  final double momentum;
  final List<Matrix64?> _velocity;

  SGD(this.parameters, {this.learningRate = 0.01, this.momentum = 0.0})
    : _velocity = List<Matrix64?>.filled(parameters.length, null);

  @override
  void step() {
    Variable.noGrad(() {
      for (var i = 0; i < parameters.length; i++) {
        final gradient = parameters[i].grad;
        if (gradient == null) {
          continue;
        }
        if (momentum > 0) {
          final previous = _velocity[i];
          final velocity = previous == null
              ? gradient
              : previous.scale(momentum) + gradient;
          _velocity[i] = velocity;
          parameters[i].value =
              parameters[i].value - velocity.scale(learningRate);
        } else {
          parameters[i].value =
              parameters[i].value - gradient.scale(learningRate);
        }
      }
    });
  }
}

/// Adam optimizer with bias-corrected first and second moment estimates.
class Adam extends Optimizer {
  @override
  final List<Variable> parameters;
  final double learningRate;
  final double beta1;
  final double beta2;
  final double epsilon;

  final List<Matrix64?> _firstMoment;
  final List<Matrix64?> _secondMoment;
  int _timestep = 0;

  Adam(
    this.parameters, {
    this.learningRate = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  }) : _firstMoment = List<Matrix64?>.filled(parameters.length, null),
       _secondMoment = List<Matrix64?>.filled(parameters.length, null);

  @override
  void step() {
    _timestep++;
    final firstCorrection = 1 - math.pow(beta1, _timestep).toDouble();
    final secondCorrection = 1 - math.pow(beta2, _timestep).toDouble();
    Variable.noGrad(() {
      for (var i = 0; i < parameters.length; i++) {
        final gradient = parameters[i].grad;
        if (gradient == null) {
          continue;
        }
        final previousFirst = _firstMoment[i];
        final first = previousFirst == null
            ? gradient.scale(1 - beta1)
            : previousFirst.scale(beta1) + gradient.scale(1 - beta1);

        final gradientSquared = gradient.hadamard(gradient);
        final previousSecond = _secondMoment[i];
        final second = previousSecond == null
            ? gradientSquared.scale(1 - beta2)
            : previousSecond.scale(beta2) + gradientSquared.scale(1 - beta2);

        _firstMoment[i] = first;
        _secondMoment[i] = second;

        final firstHat = first.scale(1 / firstCorrection);
        final secondHat = second.scale(1 / secondCorrection);
        final update = firstHat.hadamard(
          secondHat.mapValues((value) => 1.0 / (math.sqrt(value) + epsilon)),
        );
        parameters[i].value = parameters[i].value - update.scale(learningRate);
      }
    });
  }
}
