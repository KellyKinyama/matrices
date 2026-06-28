/// Script-friendly dense and sparse matrix algebra for Dart and Flutter.
///
/// The public API exposes explicit `Matrix32` and `Matrix64` precision,
/// typed-data backed vectors, direct decompositions, iterative solvers,
/// Krylov methods, SVD/PCA utilities, and CSR sparse matrices without native
/// runtime dependencies.
library matrices;

export 'src/types.dart';
export 'src/vector.dart';
export 'src/vector32.dart';
export 'src/matrix.dart';
export 'src/matrix32.dart';
export 'src/square_matrix.dart';
export 'src/decompositions.dart';
export 'src/iterative.dart';
export 'src/krylov.dart';
export 'src/advanced.dart';
export 'src/sparse_matrix.dart';
