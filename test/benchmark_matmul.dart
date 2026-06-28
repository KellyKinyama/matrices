import 'dart:io';
import 'dart:math' as math;

import 'package:matrices/matrices.dart' as matrices;
import 'package:ml_linalg/dtype.dart' as ml_dtype;
import 'package:ml_linalg/matrix.dart' as ml;

Future<void> main(List<String> args) async {
  if (args.isNotEmpty && args.first == '--runner') {
    _runReport(args.sublist(1), runtimeMode: BenchmarkRuntime.aot);
    return;
  }

  final parsedArgs = args.toList(growable: true);
  final jit = parsedArgs.remove('--jit');
  parsedArgs.remove('--aot');
  if (jit) {
    _runReport(parsedArgs, runtimeMode: BenchmarkRuntime.jit);
    return;
  }

  final runner = await _compileAotRunner(
    Platform.script.toFilePath(),
    'benchmark_matmul_runner',
  );
  final process = await Process.run(runner, [
    '--runner',
    ...parsedArgs,
  ], workingDirectory: Directory.current.path);
  stdout.write(process.stdout);
  stderr.write(process.stderr);
  if (process.exitCode != 0) {
    throw StateError(
      'AOT benchmark runner failed with exit code ${process.exitCode}',
    );
  }
}

void _runReport(List<String> args, {required BenchmarkRuntime runtimeMode}) {
  final sizes = args.isEmpty
      ? [256, 512, 1024]
      : args.first.split(',').map(int.parse).toList(growable: false);
  final iterations = args.length < 2 ? 3 : int.parse(args[1]);
  final warmups = args.length < 3 ? 1 : int.parse(args[2]);
  final reportPath = args.length < 4 ? null : args[3];
  final samples = args.length < 5 ? 3 : int.parse(args[4]);

  final results = <BenchmarkResult>[];
  for (final size in sizes) {
    results.add(_runFloat64(size, iterations, warmups, samples));
    results.add(_runFloat32(size, iterations, warmups, samples));
  }

  final command = [
    'dart run test/benchmark_matmul.dart',
    ...args,
  ].join(' ').trim();
  final markdown = _toMarkdown(
    results,
    iterations,
    warmups,
    samples,
    command,
    runtimeMode: runtimeMode,
  );
  print(markdown);
  if (reportPath != null) {
    final file = File(reportPath);
    file.parent.createSync(recursive: true);
    file.writeAsStringSync(markdown);
  }
}

BenchmarkResult _runFloat64(
  int size,
  int iterations,
  int warmups,
  int samples,
) {
  final rows = _randomRows(size, size, seed: 1);
  final otherRows = _randomRows(size, size, seed: 2);
  final oursA = matrices.Matrix64.fromRows(rows);
  final oursB = matrices.Matrix64.fromRows(otherRows);
  final mlA = ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float64);
  final mlB = ml.Matrix.fromList(otherRows, dtype: ml_dtype.DType.float64);

  final oursCheck = oursA * oursB;
  final mlCheck = mlA * mlB;
  _checkClose(_consumeOurs64(oursCheck), _consumeMl(mlCheck), 1e-8);

  for (var i = 0; i < warmups; i++) {
    _blackHole += _consumeOurs64(oursA * oursB);
    _blackHole += _consumeMl(mlA * mlB);
  }

  final oursSamples = <double>[];
  final theirsSamples = <double>[];
  for (var sample = 0; sample < samples; sample++) {
    final oursFirst = _time(
      () => _blackHole += _consumeOurs64(oursA * oursB),
      iterations,
    );
    final theirsSecond = _time(
      () => _blackHole += _consumeMl(mlA * mlB),
      iterations,
    );
    final theirsFirst = _time(
      () => _blackHole += _consumeMl(mlA * mlB),
      iterations,
    );
    final oursSecond = _time(
      () => _blackHole += _consumeOurs64(oursA * oursB),
      iterations,
    );
    oursSamples.add(_averageMsPerOp(oursFirst, oursSecond, iterations));
    theirsSamples.add(_averageMsPerOp(theirsFirst, theirsSecond, iterations));
  }
  return BenchmarkResult(
    size: size,
    precision: 'float64',
    oursInterface: 'matrices.Matrix64.fromRows + operator *',
    theirsInterface:
        'ml_linalg.Matrix64.fromList(dtype: DType.float64) + operator *',
    oursMs: _median(oursSamples),
    theirsMs: _median(theirsSamples),
  );
}

BenchmarkResult _runFloat32(
  int size,
  int iterations,
  int warmups,
  int samples,
) {
  final rows = _randomRows(size, size, seed: 1);
  final otherRows = _randomRows(size, size, seed: 2);
  final oursA = matrices.Matrix32.fromRows(rows);
  final oursB = matrices.Matrix32.fromRows(otherRows);
  final mlA = ml.Matrix.fromList(rows, dtype: ml_dtype.DType.float32);
  final mlB = ml.Matrix.fromList(otherRows, dtype: ml_dtype.DType.float32);

  final oursCheck = oursA * oursB;
  final mlCheck = mlA * mlB;
  _checkClose(_consumeOurs32(oursCheck), _consumeMl(mlCheck), 1e-2);

  for (var i = 0; i < warmups; i++) {
    _blackHole += _consumeOurs32(oursA * oursB);
    _blackHole += _consumeMl(mlA * mlB);
  }

  final oursSamples = <double>[];
  final theirsSamples = <double>[];
  for (var sample = 0; sample < samples; sample++) {
    final oursFirst = _time(
      () => _blackHole += _consumeOurs32(oursA * oursB),
      iterations,
    );
    final theirsSecond = _time(
      () => _blackHole += _consumeMl(mlA * mlB),
      iterations,
    );
    final theirsFirst = _time(
      () => _blackHole += _consumeMl(mlA * mlB),
      iterations,
    );
    final oursSecond = _time(
      () => _blackHole += _consumeOurs32(oursA * oursB),
      iterations,
    );
    oursSamples.add(_averageMsPerOp(oursFirst, oursSecond, iterations));
    theirsSamples.add(_averageMsPerOp(theirsFirst, theirsSecond, iterations));
  }
  return BenchmarkResult(
    size: size,
    precision: 'float32',
    oursInterface: 'matrices.Matrix32.fromRows + operator *',
    theirsInterface:
        'ml_linalg.Matrix64.fromList(dtype: DType.float32) + operator *',
    oursMs: _median(oursSamples),
    theirsMs: _median(theirsSamples),
  );
}

List<List<double>> _randomRows(int rows, int columns, {required int seed}) {
  final random = math.Random(seed);
  return List<List<double>>.generate(
    rows,
    (_) => List<double>.generate(columns, (_) => random.nextDouble()),
  );
}

Duration _time(void Function() run, int iterations) {
  final stopwatch = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    run();
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

String _toMarkdown(
  List<BenchmarkResult> results,
  int iterations,
  int warmups,
  int samples,
  String command, {
  required BenchmarkRuntime runtimeMode,
}) {
  final buffer = StringBuffer()
    ..writeln('# Matrix64 Multiplication Benchmark')
    ..writeln()
    ..writeln('- Command: `$command`')
    ..writeln(
      '- Runtime mode: `${runtimeMode == BenchmarkRuntime.aot ? 'AOT executable' : 'JIT VM'}`',
    )
    ..writeln('- Warmups: `$warmups`')
    ..writeln('- Samples per row: `$samples` median sample(s)')
    ..writeln(
      '- Measured iterations per sample: `$iterations x 2` timing passes',
    )
    ..writeln(
      '- Timing order: `matrices -> ml_linalg -> ml_linalg -> matrices`',
    )
    ..writeln(
      '- Input: identical deterministic random row data for both libraries',
    )
    ..writeln('- Native/FFI/BLAS: not used by `matrices`')
    ..writeln()
    ..writeln(
      '| Precision | Size | matrices interface | ml_linalg interface | matrices ms/op | ml_linalg ms/op | Speedup |',
    )
    ..writeln('|---|---:|---|---|---:|---:|---:|');
  for (final result in results) {
    buffer.writeln(
      '| ${result.precision} | ${result.size}x${result.size} | '
      '`${result.oursInterface}` | `${result.theirsInterface}` | '
      '${result.oursMs.toStringAsFixed(3)} | '
      '${result.theirsMs.toStringAsFixed(3)} | '
      '${result.speedup.toStringAsFixed(2)}x |',
    );
  }
  buffer
    ..writeln()
    ..writeln('## Interpretation')
    ..writeln()
    ..writeln(
      '- Use `Matrix64` (or `Matrix64`) for float64 numerical work where precision is the priority.',
    )
    ..writeln(
      '- Use `Matrix32` for large throughput-oriented workloads where memory bandwidth and SIMD lane width matter more than double precision.',
    )
    ..writeln(
      '- Float32 and float64 results are reported separately because they use different storage, SIMD lane width, and `ml_linalg` dtype settings.',
    );
  return buffer.toString();
}

void _checkClose(double actual, double expected, double tolerance) {
  final delta = (actual - expected).abs();
  if (delta > tolerance) {
    throw StateError('Result mismatch: $actual vs $expected (delta $delta).');
  }
}

double _consumeOurs64(matrices.Matrix64 matrix) => matrix(0, 0);

double _consumeOurs32(matrices.Matrix32 matrix) => matrix(0, 0);

double _consumeMl(ml.Matrix matrix) => matrix[0][0];

double _blackHole = 0;

enum BenchmarkRuntime { aot, jit }

class BenchmarkResult {
  final int size;
  final String precision;
  final String oursInterface;
  final String theirsInterface;
  final double oursMs;
  final double theirsMs;

  const BenchmarkResult({
    required this.size,
    required this.precision,
    required this.oursInterface,
    required this.theirsInterface,
    required this.oursMs,
    required this.theirsMs,
  });

  double get speedup => theirsMs / oursMs;
}
