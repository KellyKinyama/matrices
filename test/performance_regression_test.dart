import 'dart:io';

import 'package:test/test.dart';

void main() {
  final enabled = Platform.environment['MATRICES_PERF_REGRESSION'] == '1';

  group(
    'performance regression',
    () {
      test(
        'AOT benchmark suite meets selected thresholds',
        () async {
          final sizes =
              Platform.environment['MATRICES_PERF_SUITE_SIZES'] ?? '256';
          final iterations =
              Platform.environment['MATRICES_PERF_ITERATIONS'] ?? '3';
          final warmups = Platform.environment['MATRICES_PERF_WARMUPS'] ?? '1';
          final samples = Platform.environment['MATRICES_PERF_SAMPLES'] ?? '3';
          final reportPath =
              Platform.environment['MATRICES_PERF_REPORT'] ??
              'build/performance_regression_suite.md';

          final process = await Process.run(Platform.resolvedExecutable, [
            'run',
            'test/benchmark_suite.dart',
            sizes,
            iterations,
            warmups,
            reportPath,
            samples,
          ], workingDirectory: Directory.current.path);

          expect(
            process.exitCode,
            0,
            reason:
                'AOT benchmark suite failed.\nstdout:\n${process.stdout}\n'
                'stderr:\n${process.stderr}',
          );

          final rows = _parseBenchmarkRows(process.stdout as String);
          final size = sizes.split(',').last.trim();

          _expectRatio(
            rows,
            category: 'dense',
            operation: 'matmul',
            precision: 'float64',
            size: '${size}x$size',
            minRatio: _envDouble('MATRICES_PERF_MATMUL64_MIN_SPEEDUP', 1.05),
          );
          _expectRatio(
            rows,
            category: 'dense',
            operation: 'matmul',
            precision: 'float32',
            size: '${size}x$size',
            minRatio: _envDouble('MATRICES_PERF_MATMUL32_MIN_SPEEDUP', 1.05),
          );
          _expectRatio(
            rows,
            category: 'linear solve',
            operation: 'solve',
            precision: 'float64',
            size: '${size}x$size',
            minRatio: _envDouble('MATRICES_PERF_SOLVE64_MIN_SPEEDUP', 1.05),
          );
          _expectRatio(
            rows,
            category: 'sparse',
            operation: 'mv',
            precision: 'float64',
            sizePrefix: '2000 x 2000',
            minRatio: _envDouble('MATRICES_PERF_SPARSE_MIN_SPEEDUP', 2.00),
          );
        },
        timeout: const Timeout(Duration(minutes: 5)),
      );
    },
    skip: enabled
        ? false
        : 'Set MATRICES_PERF_REGRESSION=1 to run AOT performance thresholds.',
  );
}

List<BenchmarkRow> _parseBenchmarkRows(String markdown) {
  return markdown
      .split('\n')
      .where((line) => line.startsWith('| '))
      .skip(2)
      .map((line) => line.split('|').map((cell) => cell.trim()).toList())
      .where((cells) => cells.length >= 10)
      .map(
        (cells) => BenchmarkRow(
          category: cells[1],
          operation: cells[2],
          precision: cells[3],
          size: cells[4],
          ratio: double.tryParse(cells[9].replaceAll('x', '')),
        ),
      )
      .where((row) => row.ratio != null)
      .toList(growable: false);
}

void _expectRatio(
  List<BenchmarkRow> rows, {
  required String category,
  required String operation,
  required String precision,
  required double minRatio,
  String? size,
  String? sizePrefix,
}) {
  final matches = rows.where(
    (row) =>
        row.category == category &&
        row.operation == operation &&
        row.precision == precision &&
        (size == null || row.size == size) &&
        (sizePrefix == null || row.size.startsWith(sizePrefix)),
  );
  expect(
    matches,
    isNotEmpty,
    reason: 'Missing benchmark row for $category/$operation/$precision.',
  );
  final ratio = matches.first.ratio!;
  expect(
    ratio,
    greaterThanOrEqualTo(minRatio),
    reason:
        'Expected $category/$operation/$precision to stay at least '
        '${minRatio}x faster in AOT; actual ratio was '
        '${ratio.toStringAsFixed(2)}x.',
  );
}

double _envDouble(String name, double defaultValue) {
  return double.parse(Platform.environment[name] ?? '$defaultValue');
}

class BenchmarkRow {
  final String category;
  final String operation;
  final String precision;
  final String size;
  final double? ratio;

  const BenchmarkRow({
    required this.category,
    required this.operation,
    required this.precision,
    required this.size,
    required this.ratio,
  });
}
