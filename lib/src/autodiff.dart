import 'dart:math' as math;
import 'dart:typed_data';

import 'matrix.dart';
import 'types.dart';
import 'vector.dart';

/// CPU reverse-mode automatic differentiation node.
///
/// A [Variable] wraps a [Matrix64] [value] and, when it participates in
/// a differentiable expression, records a backward closure plus its
/// parents so that [backward] can walk the graph in reverse topological
/// order and accumulate gradients into every leaf's [grad].
///
/// The engine is pure Dart on top of the existing SIMD [Matrix64]
/// kernels — no FFI, no GPU. Gradient math is expressed with the same
/// matrix ops as the forward pass. Scalars are represented as `1x1`
/// matrices so a single code path covers losses and tensors.
///
/// Values are treated as immutable: every op allocates a fresh
/// [Matrix64] rather than mutating storage in place, so the recorded
/// graph never observes a value changing under it. Do not mutate a
/// [value] (e.g. via `set`) after it has been used in an expression.
class Variable {
  Matrix64 value;
  Matrix64? grad;
  bool requiresGrad;

  void Function()? _backward;
  List<Variable> _parents;

  Variable(this.value, {this.requiresGrad = false}) : _parents = const [];

  factory Variable.fromRows(
    Iterable<Iterable<num>> rows, {
    bool requiresGrad = false,
  }) => Variable(Matrix64.fromRows(rows), requiresGrad: requiresGrad);

  factory Variable.scalar(num value, {bool requiresGrad = false}) =>
      Variable(Matrix64.full(1, 1, value), requiresGrad: requiresGrad);

  factory Variable.filled(
    int rows,
    int columns,
    num value, {
    bool requiresGrad = false,
  }) =>
      Variable(Matrix64.full(rows, columns, value), requiresGrad: requiresGrad);

  factory Variable.zeros(int rows, int columns, {bool requiresGrad = false}) =>
      Variable(Matrix64.zeros(rows, columns), requiresGrad: requiresGrad);

  factory Variable.random(
    int rows,
    int columns, {
    int? seed,
    bool requiresGrad = false,
  }) => Variable(
    Matrix64.random(rows, columns, seed: seed),
    requiresGrad: requiresGrad,
  );

  int get rows => value.rows;

  int get columns => value.columns;

  bool get isScalar => value.rows == 1 && value.columns == 1;

  /// The single element of a `1x1` variable (e.g. a loss).
  double get item {
    if (!isScalar) {
      throw StateError(
        'item is only defined for 1x1 variables, got ($rows, $columns).',
      );
    }
    return value(0, 0);
  }

  // ---------------------------------------------------------------------
  // Autograd core.
  // ---------------------------------------------------------------------

  /// Runs reverse-mode differentiation from this node. The root is
  /// seeded with [seed], or with ones when this is a scalar. Gradients
  /// accumulate into each participating node's [grad]; call [zeroGrad]
  /// between optimization steps.
  ///
  /// When [retainGraph] is false (the default) every intermediate
  /// (non-leaf) node's forward value and gradient are released as its
  /// backward closure finishes, and the graph links are cleared. This
  /// keeps peak memory flat across training steps — only leaves
  /// (parameters and inputs) and this root's value survive. Pass
  /// `retainGraph: true` to inspect intermediates or to call
  /// [backward] again on the same graph.
  void backward({Matrix64? seed, bool retainGraph = false}) {
    if (!requiresGrad) {
      throw StateError(
        'backward() called on a Variable with requiresGrad = false.',
      );
    }
    if (seed != null) {
      if (seed.rows != rows || seed.columns != columns) {
        throw ArgumentError(
          'Seed shape (${seed.rows}, ${seed.columns}) does not match '
          '($rows, $columns).',
        );
      }
      grad = seed;
    } else if (isScalar) {
      grad = Matrix64.full(1, 1, 1.0);
    } else {
      throw StateError(
        'backward() on a non-scalar Variable requires an explicit seed gradient.',
      );
    }

    final ordered = <Variable>[];
    final visited = <Variable>{};
    void visit(Variable node) {
      if (!visited.add(node)) {
        return;
      }
      for (final parent in node._parents) {
        visit(parent);
      }
      ordered.add(node);
    }

    visit(this);
    for (var i = ordered.length - 1; i >= 0; i--) {
      final node = ordered[i];
      node._backward?.call();
      if (!retainGraph && node._backward != null) {
        // Op output: no backward closure reads a node's own forward
        // value, and all of its consumers ran earlier in reverse
        // order, so the buffers are safe to drop now.
        if (!identical(node, this)) {
          node.value = _released;
          node.grad = null;
        }
        node._parents = const [];
        node._backward = null;
      }
    }
  }

  /// Clears the accumulated gradient on this node.
  void zeroGrad() => grad = null;

  /// Returns a detached copy that does not participate in autograd.
  Variable detach() => Variable(value.copy());

  static int _noGradDepth = 0;

  /// Runs [body] with autograd disabled so no graph is recorded.
  static T noGrad<T>(T Function() body) {
    _noGradDepth++;
    try {
      return body();
    } finally {
      _noGradDepth--;
    }
  }

  void _setBackward(List<Variable> parents, void Function() backward) {
    if (_noGradDepth > 0) {
      return;
    }
    _parents = parents;
    _backward = backward;
    requiresGrad = true;
  }

  void _accumulate(Matrix64 contribution) {
    final current = grad;
    grad = current == null ? contribution : current + contribution;
  }

  /// Empty matrix that replaces a freed intermediate's forward buffer.
  static final Matrix64 _released = Matrix64.zeros(0, 0);

  // ---------------------------------------------------------------------
  // Differentiable ops.
  // ---------------------------------------------------------------------

  /// Matrix product. For `C = A @ B`: `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`.
  Variable matmul(Variable other) {
    final out = Variable(value.matmul(other.value));
    if (requiresGrad || other.requiresGrad) {
      final a = this;
      final b = other;
      out._setBackward([a, b], () {
        final g = out.grad!;
        if (a.requiresGrad) {
          a._accumulate(g.matmul(b.value.transpose));
        }
        if (b.requiresGrad) {
          b._accumulate(a.value.transpose.matmul(g));
        }
      });
    }
    return out;
  }

  Variable operator +(Variable other) {
    _checkSameShape(other);
    final out = Variable(value + other.value);
    if (requiresGrad || other.requiresGrad) {
      final a = this;
      final b = other;
      out._setBackward([a, b], () {
        final g = out.grad!;
        if (a.requiresGrad) {
          a._accumulate(g);
        }
        if (b.requiresGrad) {
          b._accumulate(g);
        }
      });
    }
    return out;
  }

  Variable operator -(Variable other) {
    _checkSameShape(other);
    final out = Variable(value - other.value);
    if (requiresGrad || other.requiresGrad) {
      final a = this;
      final b = other;
      out._setBackward([a, b], () {
        final g = out.grad!;
        if (a.requiresGrad) {
          a._accumulate(g);
        }
        if (b.requiresGrad) {
          b._accumulate(g.scale(-1));
        }
      });
    }
    return out;
  }

  /// Elementwise (Hadamard) product. `operator *` is elementwise;
  /// use [matmul] for the matrix product.
  Variable operator *(Variable other) {
    _checkSameShape(other);
    final out = Variable(value.hadamard(other.value));
    if (requiresGrad || other.requiresGrad) {
      final a = this;
      final b = other;
      out._setBackward([a, b], () {
        final g = out.grad!;
        if (a.requiresGrad) {
          a._accumulate(g.hadamard(b.value));
        }
        if (b.requiresGrad) {
          b._accumulate(g.hadamard(a.value));
        }
      });
    }
    return out;
  }

  /// Scalar multiply.
  Variable scale(num factor) {
    final s = factor.toDouble();
    final out = Variable(value.scale(s));
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () => a._accumulate(out.grad!.scale(s)));
    }
    return out;
  }

  Variable get transpose {
    final out = Variable(value.transpose);
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () => a._accumulate(out.grad!.transpose));
    }
    return out;
  }

  Variable relu() {
    final out = Variable(value.mapValues((v) => v > 0 ? v : 0.0));
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        final mask = a.value.mapValues((v) => v > 0 ? 1.0 : 0.0);
        a._accumulate(out.grad!.hadamard(mask));
      });
    }
    return out;
  }

  Variable sigmoid() {
    final s = value.mapValues((v) => 1.0 / (1.0 + math.exp(-v)));
    final out = Variable(s);
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        final derivative = s.hadamard(s.mapValues((v) => 1.0 - v));
        a._accumulate(out.grad!.hadamard(derivative));
      });
    }
    return out;
  }

  Variable tanh() {
    final t = value.mapValues(_tanh);
    final out = Variable(t);
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        final derivative = t.mapValues((v) => 1.0 - v * v);
        a._accumulate(out.grad!.hadamard(derivative));
      });
    }
    return out;
  }

  Variable exp() {
    final e = value.exp();
    final out = Variable(e);
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () => a._accumulate(out.grad!.hadamard(e)));
    }
    return out;
  }

  Variable log() {
    final out = Variable(value.mapValues(math.log));
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        final inverse = a.value.mapValues((v) => 1.0 / v);
        a._accumulate(out.grad!.hadamard(inverse));
      });
    }
    return out;
  }

  /// Sum of all elements, returning a `1x1` variable.
  Variable sum() {
    final out = Variable(Matrix64.full(1, 1, value.sum));
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        a._accumulate(Matrix64.full(a.rows, a.columns, out.grad!(0, 0)));
      });
    }
    return out;
  }

  /// Mean of all elements, returning a `1x1` variable.
  Variable mean() {
    final n = value.count;
    final out = Variable(Matrix64.full(1, 1, value.sum / n));
    if (requiresGrad) {
      final a = this;
      out._setBackward([a], () {
        a._accumulate(Matrix64.full(a.rows, a.columns, out.grad!(0, 0) / n));
      });
    }
    return out;
  }

  /// Adds a `1 x columns` row [bias] to every row (broadcast across
  /// rows). The bias gradient reduces by summing the incoming gradient
  /// over rows — the canonical linear-layer bias rule.
  Variable addRowVector(Variable bias) {
    if (bias.rows != 1 || bias.columns != columns) {
      throw ArgumentError(
        'Row bias must be (1, $columns), got (${bias.rows}, ${bias.columns}).',
      );
    }
    final biasVector = Vector64.storage(
      bias.value.unsafeValuesView,
      copy: false,
    );
    final out = Variable(value.addVector(biasVector, Axis.columns));
    if (requiresGrad || bias.requiresGrad) {
      final a = this;
      final b = bias;
      out._setBackward([a, b], () {
        final g = out.grad!;
        if (a.requiresGrad) {
          a._accumulate(g);
        }
        if (b.requiresGrad) {
          b._accumulate(_sumRows(g));
        }
      });
    }
    return out;
  }

  void _checkSameShape(Variable other) {
    if (rows != other.rows || columns != other.columns) {
      throw ArgumentError(
        'Shape mismatch: ($rows, $columns) vs '
        '(${other.rows}, ${other.columns}).',
      );
    }
  }

  @override
  String toString() =>
      'Variable(($rows, $columns), requiresGrad: $requiresGrad)';
}

Matrix64 _sumRows(Matrix64 matrix) {
  final columns = matrix.columns;
  final data = Float64List(columns);
  final flat = matrix.unsafeValuesView;
  for (var row = 0; row < matrix.rows; row++) {
    final offset = row * columns;
    for (var column = 0; column < columns; column++) {
      data[column] += flat[offset + column];
    }
  }
  return Matrix64.storage(1, columns, data, copy: false);
}

double _tanh(double x) {
  if (x > 20) {
    return 1.0;
  }
  if (x < -20) {
    return -1.0;
  }
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}
