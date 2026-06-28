import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:matrices/matrices.dart' as matrices;
import 'package:ml_linalg/dtype.dart' as ml_dtype;
import 'package:ml_linalg/matrix.dart' as ml;
import 'package:ml_linalg/vector.dart' as ml_vector;

const _variantCount = 3;
int _measurementSamples = 3;

Future<void> main(List<String> args) async {
  if (args.isNotEmpty && args.first == '--case') {
    _runSingleCase(args);
    return;
  }

  final parsedArgs = args.toList(growable: true);
  final inProcess = parsedArgs.remove('--in-process');
  final jit = parsedArgs.remove('--jit');
  parsedArgs.remove('--aot');
  final sizes = parsedArgs.isEmpty
      ? [100, 256, 512, 1000]
      : _parseSizes(parsedArgs.first);
  final iterations = parsedArgs.length < 2 ? 3 : int.parse(parsedArgs[1]);
  final warmups = parsedArgs.length < 3 ? 1 : int.parse(parsedArgs[2]);
  final reportPath = parsedArgs.length < 4
      ? 'build/benchmark_suite.md'
      : parsedArgs[3];
  _measurementSamples = parsedArgs.length < 5 ? 3 : int.parse(parsedArgs[4]);

  if (iterations < 1) {
    throw ArgumentError.value(iterations, 'iterations', 'must be >= 1');
  }
  if (warmups < 0) {
    throw ArgumentError.value(warmups, 'warmups', 'must be >= 0');
  }
  if (_measurementSamples < 1) {
    throw ArgumentError.value(_measurementSamples, 'samples', 'must be >= 1');
  }

  final cases = _benchmarkCases(sizes);
  final results = inProcess
      ? cases.map((benchmark) => benchmark.run(iterations, warmups)).toList()
      : await _runIsolatedCases(
          cases,
          sizes,
          iterations,
          warmups,
          samples: _measurementSamples,
          runtimeMode: jit ? BenchmarkRuntime.jit : BenchmarkRuntime.aot,
        );

  final command = ['dart run test/benchmark_suite.dart', ...args].join(' ');
  final markdown = _toMarkdown(
    results,
    command,
    iterations,
    warmups,
    _measurementSamples,
    isolated: !inProcess,
    runtimeMode: inProcess
        ? BenchmarkRuntime.jit
        : jit
        ? BenchmarkRuntime.jit
        : BenchmarkRuntime.aot,
  );
  print(markdown);
  final file = File(reportPath);
  file.parent.createSync(recursive: true);
  file.writeAsStringSync(markdown);
}

List<BenchmarkCase> _benchmarkCases(List<int> sizes) {
  final cases = <BenchmarkCase>[];
  for (final size in sizes) {
    cases
      ..add(
        BenchmarkCase(
          'construct64_$size',
          (iterations, warmups) => _benchConstruct64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'construct32_$size',
          (iterations, warmups) => _benchConstruct32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'scalar64_$size',
          (iterations, warmups) => _benchScalar64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'scalar32_$size',
          (iterations, warmups) => _benchScalar32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'elementwise64_$size',
          (iterations, warmups) =>
              _benchElementwise64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'elementwise32_$size',
          (iterations, warmups) =>
              _benchElementwise32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'broadcast64_$size',
          (iterations, warmups) => _benchBroadcast64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'broadcast32_$size',
          (iterations, warmups) => _benchBroadcast32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'matmul64_$size',
          (iterations, warmups) => _benchMatmul64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'matmul32_$size',
          (iterations, warmups) => _benchMatmul32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'matvec64_$size',
          (iterations, warmups) =>
              _benchMatrixVector64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'matvec32_$size',
          (iterations, warmups) =>
              _benchMatrixVector32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'vecmat64_$size',
          (iterations, warmups) =>
              _benchVectorMatrix64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'vecmat32_$size',
          (iterations, warmups) =>
              _benchVectorMatrix32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'transpose64_$size',
          (iterations, warmups) => _benchTranspose64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'transpose32_$size',
          (iterations, warmups) => _benchTranspose32(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'solve64_$size',
          (iterations, warmups) => _benchSolve64(size, iterations, warmups),
        ),
      )
      ..add(
        BenchmarkCase(
          'solve32_$size',
          (iterations, warmups) => _benchSolve32(size, iterations, warmups),
        ),
      );
  }

  final largest = sizes.reduce(math.max);
  cases
    ..add(
      BenchmarkCase(
        'vector_add64_1048576',
        (iterations, warmups) =>
            _benchVectorAdd64(1 << 20, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'vector_add32_1048576',
        (iterations, warmups) =>
            _benchVectorAdd32(1 << 20, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'vector_dot64_1048576',
        (iterations, warmups) =>
            _benchVectorDot64(1 << 20, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'vector_dot32_1048576',
        (iterations, warmups) =>
            _benchVectorDot32(1 << 20, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'rect_matmul64_$largest',
        (iterations, warmups) =>
            _benchRectangularMatmul64(largest, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'rect_matmul32_$largest',
        (iterations, warmups) =>
            _benchRectangularMatmul32(largest, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'sparse_mv_2000_0p0015',
        (iterations, warmups) =>
            _benchSparseMv(2000, 0.0015, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'sparse_mv_2000_0p01',
        (iterations, warmups) =>
            _benchSparseMv(2000, 0.01, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'lu64_128',
        (iterations, warmups) => _benchLu64(128, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'qr64_128',
        (iterations, warmups) => _benchQr64(128, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'cholesky64_128',
        (iterations, warmups) => _benchCholesky64(128, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'eigen64_64',
        (iterations, warmups) => _benchEigen64(64, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'svd64_64',
        (iterations, warmups) => _benchSvd64(64, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'svd32_64',
        (iterations, warmups) => _benchSvd32(64, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'least_squares64_256x48',
        (iterations, warmups) =>
            _benchLeastSquares64(256, 48, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'pseudo_inverse64_64',
        (iterations, warmups) => _benchPseudoInverse64(64, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'pca64_256x16',
        (iterations, warmups) => _benchPca64(256, 16, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'pca32_256x16',
        (iterations, warmups) => _benchPca32(256, 16, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'cg64_512',
        (iterations, warmups) =>
            _benchConjugateGradient64(512, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'gmres64_128',
        (iterations, warmups) => _benchGmres64(128, iterations, warmups),
      ),
    )
    ..add(
      BenchmarkCase(
        'power_iteration64_256',
        (iterations, warmups) =>
            _benchPowerIteration64(256, iterations, warmups),
      ),
    );
  return cases;
}

List<int> _parseSizes(String csv) {
  final sizes = csv.split(',').map(int.parse).toList(growable: false);
  if (sizes.isEmpty || sizes.any((size) => size < 1)) {
    throw ArgumentError.value(csv, 'sizes', 'must contain positive integers');
  }
  return sizes;
}

void _runSingleCase(List<String> args) {
  if (args.length != 6) {
    throw ArgumentError(
      'Usage: dart test/benchmark_suite.dart --case <case-id> '
      '<sizes-csv> <iterations> <warmups> <samples>',
    );
  }
  final caseId = args[1];
  final sizes = _parseSizes(args[2]);
  final iterations = int.parse(args[3]);
  final warmups = int.parse(args[4]);
  _measurementSamples = int.parse(args[5]);
  final cases = {
    for (final benchmark in _benchmarkCases(sizes)) benchmark.id: benchmark,
  };
  final benchmark = cases[caseId];
  if (benchmark == null) {
    throw ArgumentError.value(caseId, 'case-id', 'unknown benchmark case');
  }
  final result = benchmark.run(iterations, warmups);
  stdout.writeln(jsonEncode(result.toJson()));
}

Future<List<BenchmarkRow>> _runIsolatedCases(
  List<BenchmarkCase> cases,
  List<int> sizes,
  int iterations,
  int warmups, {
  required int samples,
  required BenchmarkRuntime runtimeMode,
}) async {
  final results = <BenchmarkRow>[];
  final script = Platform.script.toFilePath();
  final runner = runtimeMode == BenchmarkRuntime.aot
      ? await _compileAotRunner(script, 'benchmark_suite_runner')
      : Platform.resolvedExecutable;
  final sizesCsv = sizes.join(',');
  for (var i = 0; i < cases.length; i++) {
    final benchmark = cases[i];
    stderr.writeln('[benchmark] ${i + 1}/${cases.length} ${benchmark.id}');
    final childArgs = runtimeMode == BenchmarkRuntime.aot
        ? [
            '--case',
            benchmark.id,
            sizesCsv,
            '$iterations',
            '$warmups',
            '$samples',
          ]
        : [
            script,
            '--case',
            benchmark.id,
            sizesCsv,
            '$iterations',
            '$warmups',
            '$samples',
          ];
    final process = await Process.run(
      runner,
      childArgs,
      workingDirectory: Directory.current.path,
    );
    if (process.exitCode != 0) {
      throw StateError(
        'Benchmark case ${benchmark.id} failed with exit code '
        '${process.exitCode}\nstdout:\n${process.stdout}\nstderr:\n'
        '${process.stderr}',
      );
    }
    final lines = (process.stdout as String)
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList(growable: false);
    if (lines.isEmpty) {
      throw StateError('Benchmark case ${benchmark.id} produced no JSON row.');
    }
    results.add(
      BenchmarkRow.fromJson(jsonDecode(lines.last) as Map<String, Object?>),
    );
  }
  return results;
}

Future<String> _compileAotRunner(String script, String name) async {
  final output = _buildExecutablePath(name);
  final file = File(output);
  file.parent.createSync(recursive: true);
  stderr.writeln('[benchmark] compiling AOT runner $output');
  final process = await Process.run(Platform.resolvedExecutable, [
    'compile',
    'exe',
    script,
    '-o',
    output,
  ], workingDirectory: Directory.current.path);
  if (process.exitCode != 0) {
    throw StateError(
      'AOT compilation failed with exit code ${process.exitCode}\n'
      'stdout:\n${process.stdout}\nstderr:\n${process.stderr}',
    );
  }
  return file.absolute.path;
}

String _buildExecutablePath(String name) {
  final executable = Platform.isWindows ? '$name.exe' : name;
  return 'build/$executable';
}

BenchmarkRow _benchConstruct64(int size, int iterations, int warmups) {
  final rows = _randomRowSets(size, size, seed: 1);
  final mlRows = _copyRowSets(rows);
  return _compare(
    category: 'construction',
    operation: 'from_rows',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.fromRows',
    baselineInterface: 'ml_linalg Matrix.fromList(float64)',
    oursRun: (i) {
      final matrix = matrices.Matrix64.fromRows(rows[_pick(i)]);
      _blackHole += matrix(0, 0);
    },
    baselineRun: (i) {
      final matrix = ml.Matrix.fromList(
        mlRows[_pick(i)],
        dtype: ml_dtype.DType.float64,
      );
      _blackHole += matrix[0][0];
    },
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchConstruct32(int size, int iterations, int warmups) {
  final rows = _randomRowSets(size, size, seed: 11);
  final mlRows = _copyRowSets(rows);
  return _compare(
    category: 'construction',
    operation: 'from_rows',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32.fromRows',
    baselineInterface: 'ml_linalg Matrix.fromList(float32)',
    oursRun: (i) {
      final matrix = matrices.Matrix32.fromRows(rows[_pick(i)]);
      _blackHole += matrix(0, 0);
    },
    baselineRun: (i) {
      final matrix = ml.Matrix.fromList(
        mlRows[_pick(i)],
        dtype: ml_dtype.DType.float32,
      );
      _blackHole += matrix[0][0];
    },
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchScalar64(int size, int iterations, int warmups) {
  final ours = _matrices64(size, size, seed: 21);
  final baseline = _mlMatrices(
    size,
    size,
    seed: 21,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'elementwise',
    operation: 'scalar_multiply',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64 * double',
    baselineInterface: 'ml_linalg Matrix.float64 * double',
    oursRun: (i) => _blackHole += (ours[_pick(i)] * 1.0001)(0, 0),
    baselineRun: (i) => _blackHole += (baseline[_pick(i)] * 1.0001)[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchScalar32(int size, int iterations, int warmups) {
  final ours = _matrices32(size, size, seed: 31);
  final baseline = _mlMatrices(
    size,
    size,
    seed: 31,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'elementwise',
    operation: 'scalar_multiply',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32 * double',
    baselineInterface: 'ml_linalg Matrix.float32 * double',
    oursRun: (i) => _blackHole += (ours[_pick(i)] * 1.0001)(0, 0),
    baselineRun: (i) => _blackHole += (baseline[_pick(i)] * 1.0001)[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchElementwise64(int size, int iterations, int warmups) {
  final oursA = _matrices64(size, size, seed: 41);
  final oursB = _matrices64(size, size, seed: 51);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 41,
    dtype: ml_dtype.DType.float64,
  );
  final baselineB = _mlMatrices(
    size,
    size,
    seed: 51,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'elementwise',
    operation: 'add',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64 + Matrix64',
    baselineInterface: 'ml_linalg Matrix.float64 + Matrix64.float64',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] + oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] + baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchElementwise32(int size, int iterations, int warmups) {
  final oursA = _matrices32(size, size, seed: 61);
  final oursB = _matrices32(size, size, seed: 71);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 61,
    dtype: ml_dtype.DType.float32,
  );
  final baselineB = _mlMatrices(
    size,
    size,
    seed: 71,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'elementwise',
    operation: 'add',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32 + Matrix32',
    baselineInterface: 'ml_linalg Matrix.float32 + Matrix64.float32',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] + oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] + baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchBroadcast64(int size, int iterations, int warmups) {
  final matrices64 = _matrices64(size, size, seed: 81);
  final vectors = _vectors64(size, seed: 91);
  return _oursOnly(
    category: 'broadcast',
    operation: 'row_add',
    precision: 'float64',
    size: '${size}x$size + $size',
    oursInterface: 'Matrix64 + Vector64',
    baselineInterface: 'not supported by ml_linalg Matrix + Vector64',
    oursRun: (i) =>
        _blackHole += (matrices64[_pick(i)] + vectors[_pick(i)])(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchBroadcast32(int size, int iterations, int warmups) {
  final matrices32 = _matrices32(size, size, seed: 101);
  final vectors = _vectors32(size, seed: 111);
  return _oursOnly(
    category: 'broadcast',
    operation: 'row_add',
    precision: 'float32',
    size: '${size}x$size + $size',
    oursInterface: 'Matrix32 + Vector32',
    baselineInterface: 'not supported by ml_linalg Matrix + Vector64',
    oursRun: (i) =>
        _blackHole += (matrices32[_pick(i)] + vectors[_pick(i)])(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchMatmul64(int size, int iterations, int warmups) {
  final oursA = _matrices64(size, size, seed: 121);
  final oursB = _matrices64(size, size, seed: 131);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 121,
    dtype: ml_dtype.DType.float64,
  );
  final baselineB = _mlMatrices(
    size,
    size,
    seed: 131,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'dense',
    operation: 'matmul',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64 * Matrix64',
    baselineInterface: 'ml_linalg Matrix.float64 * Matrix64.float64',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchMatmul32(int size, int iterations, int warmups) {
  final oursA = _matrices32(size, size, seed: 141);
  final oursB = _matrices32(size, size, seed: 151);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 141,
    dtype: ml_dtype.DType.float32,
  );
  final baselineB = _mlMatrices(
    size,
    size,
    seed: 151,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'dense',
    operation: 'matmul',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32 * Matrix32',
    baselineInterface: 'ml_linalg Matrix.float32 * Matrix64.float32',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchRectangularMatmul64(int size, int iterations, int warmups) {
  final rows = size * 2;
  final inner = math.max(16, size ~/ 4);
  final columns = math.max(32, size ~/ 2);
  final oursA = _matrices64(rows, inner, seed: 161);
  final oursB = _matrices64(inner, columns, seed: 171);
  final baselineA = _mlMatrices(
    rows,
    inner,
    seed: 161,
    dtype: ml_dtype.DType.float64,
  );
  final baselineB = _mlMatrices(
    inner,
    columns,
    seed: 171,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'dense',
    operation: 'rectangular_matmul',
    precision: 'float64',
    size: '${rows}x$inner * ${inner}x$columns',
    oursInterface: 'Matrix64 * Matrix64',
    baselineInterface: 'ml_linalg Matrix.float64 * Matrix64.float64',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchRectangularMatmul32(int size, int iterations, int warmups) {
  final rows = size * 2;
  final inner = math.max(16, size ~/ 4);
  final columns = math.max(32, size ~/ 2);
  final oursA = _matrices32(rows, inner, seed: 181);
  final oursB = _matrices32(inner, columns, seed: 191);
  final baselineA = _mlMatrices(
    rows,
    inner,
    seed: 181,
    dtype: ml_dtype.DType.float32,
  );
  final baselineB = _mlMatrices(
    inner,
    columns,
    seed: 191,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'dense',
    operation: 'rectangular_matmul',
    precision: 'float32',
    size: '${rows}x$inner * ${inner}x$columns',
    oursInterface: 'Matrix32 * Matrix32',
    baselineInterface: 'ml_linalg Matrix.float32 * Matrix64.float32',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchMatrixVector64(int size, int iterations, int warmups) {
  final oursA = _matrices64(size, size, seed: 201);
  final oursX = _vectors64(size, seed: 211);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 201,
    dtype: ml_dtype.DType.float64,
  );
  final baselineX = _mlVectors(size, seed: 211, dtype: ml_dtype.DType.float64);
  return _compare(
    category: 'dense',
    operation: 'matvec',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64 * Vector64',
    baselineInterface: 'ml_linalg Matrix.float64 * Vector64.float64',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursX[_pick(i)])[0],
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineX[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchMatrixVector32(int size, int iterations, int warmups) {
  final oursA = _matrices32(size, size, seed: 221);
  final oursX = _vectors32(size, seed: 231);
  final baselineA = _mlMatrices(
    size,
    size,
    seed: 221,
    dtype: ml_dtype.DType.float32,
  );
  final baselineX = _mlVectors(size, seed: 231, dtype: ml_dtype.DType.float32);
  return _compare(
    category: 'dense',
    operation: 'matvec',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32 * Vector32',
    baselineInterface: 'ml_linalg Matrix.float32 * Vector64.float32',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] * oursX[_pick(i)])[0],
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] * baselineX[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorMatrix64(int size, int iterations, int warmups) {
  final vectors = _vectors64(size, seed: 241);
  final matrices64 = _matrices64(size, size, seed: 251);
  final baselineVectors = _mlVectors(
    size,
    seed: 241,
    dtype: ml_dtype.DType.float64,
  );
  final baselineMatrices = _mlMatrices(
    size,
    size,
    seed: 251,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'dense',
    operation: 'vecmat',
    precision: 'float64',
    size: '$size * ${size}x$size',
    oursInterface: 'Vector64 * Matrix64',
    baselineInterface: 'ml_linalg Vector.float64 * Matrix64.float64',
    oursRun: (i) => _blackHole += (vectors[_pick(i)] * matrices64[_pick(i)])[0],
    baselineRun: (i) => _blackHole +=
        (baselineVectors[_pick(i)] * baselineMatrices[_pick(i)])[0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorMatrix32(int size, int iterations, int warmups) {
  final vectors = _vectors32(size, seed: 261);
  final matrices32 = _matrices32(size, size, seed: 271);
  final baselineVectors = _mlVectors(
    size,
    seed: 261,
    dtype: ml_dtype.DType.float32,
  );
  final baselineMatrices = _mlMatrices(
    size,
    size,
    seed: 271,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'dense',
    operation: 'vecmat',
    precision: 'float32',
    size: '$size * ${size}x$size',
    oursInterface: 'Vector32 * Matrix32',
    baselineInterface: 'ml_linalg Vector.float32 * Matrix64.float32',
    oursRun: (i) => _blackHole += (vectors[_pick(i)] * matrices32[_pick(i)])[0],
    baselineRun: (i) => _blackHole +=
        (baselineVectors[_pick(i)] * baselineMatrices[_pick(i)])[0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchTranspose64(int size, int iterations, int warmups) {
  final ours = _matrices64(size, size, seed: 281);
  final baseline = _mlMatrices(
    size,
    size,
    seed: 281,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'dense',
    operation: 'transpose',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.transpose',
    baselineInterface: 'ml_linalg Matrix.transpose()',
    oursRun: (i) => _blackHole += ours[_pick(i)].transpose(0, 0),
    baselineRun: (i) => _blackHole += baseline[_pick(i)].transpose()[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchTranspose32(int size, int iterations, int warmups) {
  final ours = _matrices32(size, size, seed: 291);
  final baseline = _mlMatrices(
    size,
    size,
    seed: 291,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'dense',
    operation: 'transpose',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32.transpose',
    baselineInterface: 'ml_linalg Matrix.transpose()',
    oursRun: (i) => _blackHole += ours[_pick(i)].transpose(0, 0),
    baselineRun: (i) => _blackHole += baseline[_pick(i)].transpose()[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchSolve64(int size, int iterations, int warmups) {
  final aRows = _spdRowSets(size, seed: 301);
  final bRows = _randomRowSets(size, 1, seed: 311);
  final oursA = aRows.map(matrices.Matrix64.fromRows).toList(growable: false);
  final oursB = bRows.map(matrices.Matrix64.fromRows).toList(growable: false);
  final baselineA = aRows
      .map((rows) => ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float64))
      .toList(growable: false);
  final baselineB = bRows
      .map((rows) => ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float64))
      .toList(growable: false);
  return _compare(
    category: 'linear solve',
    operation: 'solve',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.solve',
    baselineInterface: 'ml_linalg Matrix.solve',
    oursRun: (i) => _blackHole += oursA[_pick(i)].solve(oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += baselineA[_pick(i)].solve(baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchSolve32(int size, int iterations, int warmups) {
  final aRows = _spdRowSets(size, seed: 321);
  final bRows = _randomRowSets(size, 1, seed: 331);
  final oursA = aRows.map(matrices.Matrix32.fromRows).toList(growable: false);
  final oursB = bRows.map(matrices.Matrix32.fromRows).toList(growable: false);
  final baselineA = aRows
      .map((rows) => ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float32))
      .toList(growable: false);
  final baselineB = bRows
      .map((rows) => ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float32))
      .toList(growable: false);
  return _compare(
    category: 'linear solve',
    operation: 'solve',
    precision: 'float32',
    size: '${size}x$size',
    oursInterface: 'Matrix32.solve',
    baselineInterface: 'ml_linalg Matrix.solve(float32)',
    oursRun: (i) => _blackHole += oursA[_pick(i)].solve(oursB[_pick(i)])(0, 0),
    baselineRun: (i) =>
        _blackHole += baselineA[_pick(i)].solve(baselineB[_pick(i)])[0][0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorAdd64(int length, int iterations, int warmups) {
  final oursA = _vectors64(length, seed: 341);
  final oursB = _vectors64(length, seed: 351);
  final baselineA = _mlVectors(
    length,
    seed: 341,
    dtype: ml_dtype.DType.float64,
  );
  final baselineB = _mlVectors(
    length,
    seed: 351,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'vector',
    operation: 'add',
    precision: 'float64',
    size: '$length',
    oursInterface: 'Vector64 + Vector64',
    baselineInterface: 'ml_linalg Vector.float64 + Vector64.float64',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] + oursB[_pick(i)])[0],
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] + baselineB[_pick(i)])[0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorAdd32(int length, int iterations, int warmups) {
  final oursA = _vectors32(length, seed: 361);
  final oursB = _vectors32(length, seed: 371);
  final baselineA = _mlVectors(
    length,
    seed: 361,
    dtype: ml_dtype.DType.float32,
  );
  final baselineB = _mlVectors(
    length,
    seed: 371,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'vector',
    operation: 'add',
    precision: 'float32',
    size: '$length',
    oursInterface: 'Vector32 + Vector32',
    baselineInterface: 'ml_linalg Vector.float32 + Vector64.float32',
    oursRun: (i) => _blackHole += (oursA[_pick(i)] + oursB[_pick(i)])[0],
    baselineRun: (i) =>
        _blackHole += (baselineA[_pick(i)] + baselineB[_pick(i)])[0],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorDot64(int length, int iterations, int warmups) {
  final oursA = _vectors64(length, seed: 381);
  final oursB = _vectors64(length, seed: 391);
  final baselineA = _mlVectors(
    length,
    seed: 381,
    dtype: ml_dtype.DType.float64,
  );
  final baselineB = _mlVectors(
    length,
    seed: 391,
    dtype: ml_dtype.DType.float64,
  );
  return _compare(
    category: 'vector',
    operation: 'dot',
    precision: 'float64',
    size: '$length',
    oursInterface: 'Vector64.dot',
    baselineInterface: 'ml_linalg Vector.float64.dot',
    oursRun: (i) => _blackHole += oursA[_pick(i)].dot(oursB[_pick(i)]),
    baselineRun: (i) =>
        _blackHole += baselineA[_pick(i)].dot(baselineB[_pick(i)]),
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchVectorDot32(int length, int iterations, int warmups) {
  final oursA = _vectors32(length, seed: 401);
  final oursB = _vectors32(length, seed: 411);
  final baselineA = _mlVectors(
    length,
    seed: 401,
    dtype: ml_dtype.DType.float32,
  );
  final baselineB = _mlVectors(
    length,
    seed: 411,
    dtype: ml_dtype.DType.float32,
  );
  return _compare(
    category: 'vector',
    operation: 'dot',
    precision: 'float32',
    size: '$length',
    oursInterface: 'Vector32.dot',
    baselineInterface: 'ml_linalg Vector.float32.dot',
    oursRun: (i) => _blackHole += oursA[_pick(i)].dot(oursB[_pick(i)]),
    baselineRun: (i) =>
        _blackHole += baselineA[_pick(i)].dot(baselineB[_pick(i)]),
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchSparseMv(
  int size,
  double density,
  int iterations,
  int warmups,
) {
  final entries = _sparseEntries(
    size,
    density,
    seed: density == 0.0015 ? 421 : 431,
  );
  final sparse = matrices.SparseMatrix.fromTriplets(size, size, entries);
  final dense = sparse.toDense();
  final vector = matrices.vec(List<double>.filled(size, 1));
  return _oursOnly(
    category: 'sparse',
    operation: 'mv',
    precision: 'float64',
    size: '$size x $size, nnz=${sparse.nnz}',
    oursInterface: 'SparseMatrix.mv',
    baselineInterface: 'Matrix64.mv dense baseline',
    oursRun: (i) => _blackHole += sparse.mv(vector)[i % size],
    baselineRun: (i) => _blackHole += dense.mv(vector)[i % size],
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchLu64(int size, int iterations, int warmups) {
  final matrices64 = _spdRowSets(
    size,
    seed: 441,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  return _oursOnly(
    category: 'decomposition',
    operation: 'lu',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.lu',
    baselineInterface: 'not benchmarked',
    oursRun: (i) => _blackHole += matrices64[_pick(i)].lu().upper(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchQr64(int size, int iterations, int warmups) {
  final matrices64 = _matrices64(size, math.max(8, size ~/ 2), seed: 451);
  return _oursOnly(
    category: 'decomposition',
    operation: 'qr',
    precision: 'float64',
    size: '$size x ${math.max(8, size ~/ 2)}',
    oursInterface: 'Matrix64.qr',
    baselineInterface: 'not benchmarked',
    oursRun: (i) => _blackHole += matrices64[_pick(i)].qr().r(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchCholesky64(int size, int iterations, int warmups) {
  final matrices64 = _spdRowSets(
    size,
    seed: 461,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  return _oursOnly(
    category: 'decomposition',
    operation: 'cholesky',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.cholesky',
    baselineInterface: 'not benchmarked',
    oursRun: (i) => _blackHole += matrices64[_pick(i)].cholesky().lower(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchEigen64(int size, int iterations, int warmups) {
  final matrices64 = _symmetricRowSets(
    size,
    seed: 471,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  return _oursOnly(
    category: 'decomposition',
    operation: 'eigen_symmetric',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.eigenSymmetric',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) =>
        _blackHole += matrices64[_pick(i)].eigenSymmetric().values[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchSvd64(int size, int iterations, int warmups) {
  final matrices64 = _matrices64(size, size ~/ 2, seed: 481);
  return _oursOnly(
    category: 'decomposition',
    operation: 'svd',
    precision: 'float64',
    size: '$size x ${size ~/ 2}',
    oursInterface: 'Matrix64.svd',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)].svd().singularValues[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchSvd32(int size, int iterations, int warmups) {
  final matrices32 = _matrices32(size, size ~/ 2, seed: 491);
  return _oursOnly(
    category: 'decomposition',
    operation: 'svd',
    precision: 'float32',
    size: '$size x ${size ~/ 2}',
    oursInterface: 'Matrix32.svd',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices32[_pick(i)].svd().singularValues[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchLeastSquares64(
  int rows,
  int columns,
  int iterations,
  int warmups,
) {
  final matrices64 = _matrices64(rows, columns, seed: 501);
  final rhs = _matrices64(rows, 2, seed: 511);
  return _oursOnly(
    category: 'least squares',
    operation: 'least_squares',
    precision: 'float64',
    size: '$rows x $columns',
    oursInterface: 'Matrix64.leastSquares',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) =>
        _blackHole += matrices64[_pick(i)].leastSquares(rhs[_pick(i)])(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchPseudoInverse64(int size, int iterations, int warmups) {
  final matrices64 = _matrices64(size, size ~/ 2, seed: 521);
  return _oursOnly(
    category: 'least squares',
    operation: 'pseudo_inverse',
    precision: 'float64',
    size: '$size x ${size ~/ 2}',
    oursInterface: 'Matrix64.pseudoInverse',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)].pseudoInverse()(0, 0),
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchPca64(int rows, int columns, int iterations, int warmups) {
  final matrices64 = _matrices64(rows, columns, seed: 531);
  return _oursOnly(
    category: 'statistics',
    operation: 'pca',
    precision: 'float64',
    size: '$rows x $columns',
    oursInterface: 'Matrix64.pca',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)]
        .pca(components: 4)
        .explainedVariance[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchPca32(int rows, int columns, int iterations, int warmups) {
  final matrices32 = _matrices32(rows, columns, seed: 541);
  return _oursOnly(
    category: 'statistics',
    operation: 'pca',
    precision: 'float32',
    size: '$rows x $columns',
    oursInterface: 'Matrix32.pca',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices32[_pick(i)]
        .pca(components: 4)
        .explainedVariance[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchConjugateGradient64(int size, int iterations, int warmups) {
  final matrices64 = _spdRowSets(
    size,
    seed: 551,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  final rhs = _vectors64(size, seed: 561);
  return _oursOnly(
    category: 'iterative',
    operation: 'conjugate_gradient',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.conjugateGradient',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)]
        .conjugateGradient(rhs[_pick(i)], maxIterations: 80)
        .solution[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchGmres64(int size, int iterations, int warmups) {
  final matrices64 = _spdRowSets(
    size,
    seed: 571,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  final rhs = _vectors64(size, seed: 581);
  return _oursOnly(
    category: 'iterative',
    operation: 'gmres',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.gmres',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)]
        .gmres(rhs[_pick(i)], restart: 16, maxIterations: 64)
        .solution[0],
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _benchPowerIteration64(int size, int iterations, int warmups) {
  final matrices64 = _symmetricRowSets(
    size,
    seed: 591,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
  return _oursOnly(
    category: 'iterative',
    operation: 'power_iteration',
    precision: 'float64',
    size: '${size}x$size',
    oursInterface: 'Matrix64.powerIteration',
    baselineInterface: 'not provided by ml_linalg',
    oursRun: (i) => _blackHole += matrices64[_pick(i)]
        .powerIteration(maxIterations: 80)
        .eigenvalue,
    baselineRun: null,
    iterations: iterations,
    warmups: warmups,
  );
}

BenchmarkRow _compare({
  required String category,
  required String operation,
  required String precision,
  required String size,
  required String oursInterface,
  required String baselineInterface,
  required void Function(int iteration) oursRun,
  required void Function(int iteration) baselineRun,
  required int iterations,
  required int warmups,
}) {
  _warm(oursRun, warmups);
  _warm(baselineRun, warmups);
  final oursSamples = <double>[];
  final baselineSamples = <double>[];
  for (var sample = 0; sample < _measurementSamples; sample++) {
    final baseIteration = sample * iterations * 4;
    final oursFirst = _time(oursRun, iterations, startIteration: baseIteration);
    final baselineSecond = _time(
      baselineRun,
      iterations,
      startIteration: baseIteration + iterations,
    );
    final baselineFirst = _time(
      baselineRun,
      iterations,
      startIteration: baseIteration + iterations * 2,
    );
    final oursSecond = _time(
      oursRun,
      iterations,
      startIteration: baseIteration + iterations * 3,
    );
    oursSamples.add(_averageMsPerOp(oursFirst, oursSecond, iterations));
    baselineSamples.add(
      _averageMsPerOp(baselineFirst, baselineSecond, iterations),
    );
  }
  return BenchmarkRow(
    category: category,
    operation: operation,
    precision: precision,
    size: size,
    oursInterface: oursInterface,
    baselineInterface: baselineInterface,
    oursMs: _median(oursSamples),
    baselineMs: _median(baselineSamples),
  );
}

BenchmarkRow _oursOnly({
  required String category,
  required String operation,
  required String precision,
  required String size,
  required String oursInterface,
  required String baselineInterface,
  required void Function(int iteration) oursRun,
  required void Function(int iteration)? baselineRun,
  required int iterations,
  required int warmups,
}) {
  _warm(oursRun, warmups);
  if (baselineRun != null) {
    _warm(baselineRun, warmups);
  }
  final oursSamples = <double>[];
  final baselineSamples = <double>[];
  for (var sample = 0; sample < _measurementSamples; sample++) {
    final baseIteration = sample * iterations * 4;
    final oursFirst = _time(oursRun, iterations, startIteration: baseIteration);
    final baselineSecond = baselineRun == null
        ? null
        : _time(
            baselineRun,
            iterations,
            startIteration: baseIteration + iterations,
          );
    final baselineFirst = baselineRun == null
        ? null
        : _time(
            baselineRun,
            iterations,
            startIteration: baseIteration + iterations * 2,
          );
    final oursSecond = _time(
      oursRun,
      iterations,
      startIteration: baseIteration + iterations * 3,
    );
    oursSamples.add(_averageMsPerOp(oursFirst, oursSecond, iterations));
    if (baselineFirst != null && baselineSecond != null) {
      baselineSamples.add(
        _averageMsPerOp(baselineFirst, baselineSecond, iterations),
      );
    }
  }
  return BenchmarkRow(
    category: category,
    operation: operation,
    precision: precision,
    size: size,
    oursInterface: oursInterface,
    baselineInterface: baselineInterface,
    oursMs: _median(oursSamples),
    baselineMs: baselineSamples.isEmpty ? null : _median(baselineSamples),
  );
}

List<matrices.Matrix64> _matrices64(
  int rows,
  int columns, {
  required int seed,
}) {
  return _randomRowSets(
    rows,
    columns,
    seed: seed,
  ).map(matrices.Matrix64.fromRows).toList(growable: false);
}

List<matrices.Matrix32> _matrices32(
  int rows,
  int columns, {
  required int seed,
}) {
  return _randomRowSets(
    rows,
    columns,
    seed: seed,
  ).map(matrices.Matrix32.fromRows).toList(growable: false);
}

List<ml.Matrix> _mlMatrices(
  int rows,
  int columns, {
  required int seed,
  required ml_dtype.DType dtype,
}) {
  return _randomRowSets(rows, columns, seed: seed)
      .map((rows) => ml.Matrix.fromList(rows, dtype: dtype))
      .toList(growable: false);
}

List<matrices.Vector64> _vectors64(int length, {required int seed}) {
  return _randomRowSets(
    1,
    length,
    seed: seed,
  ).map((rows) => matrices.Vector64(rows.single)).toList(growable: false);
}

List<matrices.Vector32> _vectors32(int length, {required int seed}) {
  return _randomRowSets(
    1,
    length,
    seed: seed,
  ).map((rows) => matrices.Vector32(rows.single)).toList(growable: false);
}

List<ml_vector.Vector> _mlVectors(
  int length, {
  required int seed,
  required ml_dtype.DType dtype,
}) {
  return _randomRowSets(1, length, seed: seed)
      .map((rows) => ml_vector.Vector.fromList(rows.single, dtype: dtype))
      .toList(growable: false);
}

List<List<List<double>>> _randomRowSets(
  int rows,
  int columns, {
  required int seed,
}) {
  return List<List<List<double>>>.generate(
    _variantCount,
    (variant) => _randomRows(rows, columns, seed: seed + variant),
    growable: false,
  );
}

List<List<List<double>>> _copyRowSets(List<List<List<double>>> source) {
  return source
      .map(
        (rows) => rows
            .map((row) => row.toList(growable: false))
            .toList(growable: false),
      )
      .toList(growable: false);
}

List<List<List<double>>> _spdRowSets(int size, {required int seed}) {
  return List<List<List<double>>>.generate(
    _variantCount,
    (variant) => _spdRows(size, seed: seed + variant),
    growable: false,
  );
}

List<List<List<double>>> _symmetricRowSets(int size, {required int seed}) {
  return List<List<List<double>>>.generate(_variantCount, (variant) {
    final a = matrices.Matrix64.random(size, size, seed: seed + variant);
    final symmetric = (a + a.transpose) / 2 + matrices.Matrix64.eye(size);
    return symmetric.toRows();
  }, growable: false);
}

List<matrices.SparseEntry> _sparseEntries(
  int size,
  double density, {
  required int seed,
}) {
  final random = math.Random(seed);
  final entries = <matrices.SparseEntry>[];
  final target = math.max(size, (size * size * density).round());
  final seen = <int>{};
  for (var i = 0; i < size; i++) {
    entries.add(matrices.SparseEntry(i, i, 2.0));
    seen.add(i * size + i);
  }
  while (entries.length < target) {
    final row = random.nextInt(size);
    final column = random.nextInt(size);
    final key = row * size + column;
    if (!seen.add(key)) {
      continue;
    }
    entries.add(matrices.SparseEntry(row, column, random.nextDouble() - 0.5));
  }
  return entries;
}

List<List<double>> _randomRows(int rows, int columns, {required int seed}) {
  final random = math.Random(seed);
  return List<List<double>>.generate(
    rows,
    (_) => List<double>.generate(columns, (_) => random.nextDouble()),
  );
}

List<List<double>> _spdRows(int size, {required int seed}) {
  final a = matrices.Matrix64.random(size, size, seed: seed);
  return (a.transpose * a + matrices.Matrix64.eye(size) * size).toRows();
}

int _pick(int iteration) => iteration % _variantCount;

void _warm(void Function(int iteration) run, int warmups) {
  for (var i = 0; i < warmups; i++) {
    run(i);
  }
}

Duration _time(
  void Function(int iteration) run,
  int iterations, {
  int startIteration = 0,
}) {
  final stopwatch = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    run(startIteration + i);
  }
  stopwatch.stop();
  return stopwatch.elapsed;
}

double _averageMsPerOp(Duration first, Duration second, int iterations) {
  return (first.inMicroseconds + second.inMicroseconds) / iterations / 2 / 1000;
}

double _median(List<double> values) {
  values.sort();
  final middle = values.length ~/ 2;
  if (values.length.isOdd) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2;
}

String _toMarkdown(
  List<BenchmarkRow> results,
  String command,
  int iterations,
  int warmups,
  int samples, {
  required bool isolated,
  required BenchmarkRuntime runtimeMode,
}) {
  final comparable = results.where((row) => row.baselineMs != null).toList();
  final faster = comparable
      .where((row) => row.baselineMs! / row.oursMs >= 1)
      .length;
  final categories = results.map((row) => row.category).toSet().toList()
    ..sort();

  final buffer = StringBuffer()
    ..writeln('# Benchmark Suite')
    ..writeln()
    ..writeln('- Command: `$command`')
    ..writeln(
      '- Execution mode: `${isolated ? 'isolated subprocess per benchmark row' : 'single VM process (--in-process)'}`',
    )
    ..writeln(
      '- Runtime mode: `${runtimeMode == BenchmarkRuntime.aot ? 'AOT executable' : 'JIT VM'}`',
    )
    ..writeln('- Warmups: `$warmups`')
    ..writeln('- Samples per row: `$samples` median sample(s)')
    ..writeln(
      '- Measured iterations per sample: `$iterations x 2` timing passes',
    )
    ..writeln('- Timing order: `matrices -> baseline -> baseline -> matrices`')
    ..writeln(
      '- Input rotation: `$_variantCount` deterministic variants per benchmark row',
    )
    ..writeln('- Native/FFI/BLAS: not used by `matrices`')
    ..writeln(
      '- Comparable rows faster or equal: `$faster / ${comparable.length}`',
    )
    ..writeln()
    ..writeln('## Coverage')
    ..writeln()
    ..writeln(categories.map((category) => '- $category').join('\n'))
    ..writeln()
    ..writeln('## Results')
    ..writeln()
    ..writeln(
      '| Category | Operation | Precision | Size | matrices interface | Baseline interface | matrices ms/op | Baseline ms/op | Ratio |',
    )
    ..writeln('|---|---|---|---:|---|---|---:|---:|---:|');
  for (final result in results) {
    final baseline = result.baselineMs;
    buffer.writeln(
      '| ${result.category} | ${result.operation} | ${result.precision} | '
      '${result.size} | `${result.oursInterface}` | '
      '`${result.baselineInterface}` | '
      '${result.oursMs.toStringAsFixed(3)} | '
      '${baseline == null ? 'n/a' : baseline.toStringAsFixed(3)} | '
      '${baseline == null ? 'n/a' : '${(baseline / result.oursMs).toStringAsFixed(2)}x'} |',
    );
  }
  buffer
    ..writeln()
    ..writeln('## Notes')
    ..writeln()
    ..writeln(
      '- Ratios above `1.00x` mean `matrices` is faster than the baseline for that row.',
    )
    ..writeln(
      '- `not provided by ml_linalg` rows measure APIs that do not have a direct ml_linalg equivalent.',
    )
    ..writeln(
      '- Sparse rows compare CSR traversal against this package\'s dense `Matrix64.mv` baseline, not against ml_linalg.',
    )
    ..writeln(
      '- Fixture construction is outside the timed region, except construction rows where matrix construction is the measured operation.',
    )
    ..writeln(
      '- Isolated mode excludes process startup from the timed region and prevents one row from inheriting another row\'s heap, GC, or warmed code state.',
    )
    ..writeln(
      '- AOT mode compiles the benchmark runner before measurement; compilation time is excluded from all row timings.',
    )
    ..writeln(
      '- Benchmarks are machine-sensitive; rerun on the target Dart SDK, CPU, and power profile before publishing claims.',
    );
  return buffer.toString();
}

double _blackHole = 0;

class BenchmarkRow {
  final String category;
  final String operation;
  final String precision;
  final String size;
  final String oursInterface;
  final String baselineInterface;
  final double oursMs;
  final double? baselineMs;

  const BenchmarkRow({
    required this.category,
    required this.operation,
    required this.precision,
    required this.size,
    required this.oursInterface,
    required this.baselineInterface,
    required this.oursMs,
    required this.baselineMs,
  });

  factory BenchmarkRow.fromJson(Map<String, Object?> json) {
    return BenchmarkRow(
      category: json['category']! as String,
      operation: json['operation']! as String,
      precision: json['precision']! as String,
      size: json['size']! as String,
      oursInterface: json['oursInterface']! as String,
      baselineInterface: json['baselineInterface']! as String,
      oursMs: (json['oursMs']! as num).toDouble(),
      baselineMs: (json['baselineMs'] as num?)?.toDouble(),
    );
  }

  Map<String, Object?> toJson() {
    return {
      'category': category,
      'operation': operation,
      'precision': precision,
      'size': size,
      'oursInterface': oursInterface,
      'baselineInterface': baselineInterface,
      'oursMs': oursMs,
      'baselineMs': baselineMs,
    };
  }
}

typedef BenchmarkRunner = BenchmarkRow Function(int iterations, int warmups);

enum BenchmarkRuntime { aot, jit }

class BenchmarkCase {
  final String id;
  final BenchmarkRunner run;

  const BenchmarkCase(this.id, this.run);
}
