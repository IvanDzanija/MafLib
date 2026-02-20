# MafLib

MafLib is a modern, header-only C++ library providing mathematical utilities, linear algebra operations, data structures, and algorithms for numerical computing applications.

## Overview

MafLib is designed as a comprehensive C++ library that combines mathematical computation capabilities with essential data structures and algorithms. The library emphasizes performance through modern C++ features, template-based design, and platform-specific optimizations.

## Core Components

### Linear Algebra
- Matrix operations with row-major dense storage
- Vector operations and utilities
- Matrix decompositions: PLU, QR, Cholesky
- Eigenvalue and eigenvector computation
- Principal Component Analysis (PCA)
- Norms and matrix/vector checkers
- Platform-optimized routines using Apple Accelerate framework (macOS) and BLAS/LAPACK

### Mathematical Utilities
- Extended integer arithmetic
- Polynomial operations and constructors
- Random variable generators
- Statistical functions
- Modular arithmetic
- Bit manipulation utilities
- Type conversions

### Data Structures
- Tree implementations: B-tree, Binary tree
- Trie for string operations
- Disjoint set (union-find)
- AVL tree (basic structure, in development)

### Parsers
- Finance CSV parser (planned)

## Key Features

- **Header-Only Design**: Simple integration into existing projects without complex build configurations
- **Modern C++20**: Leverages concepts, ranges, and other modern C++ features
- **Performance Optimized**: 
  - OpenMP parallelization for intensive computations
  - Apple Accelerate framework support on macOS
  - Cache-friendly blocking strategies
  - Compiler optimizations with Clang
- **Template-Based**: Generic programming for flexibility across numeric types
- **Well-Tested**: Comprehensive test suite covering core functionality

## Requirements

- **Compiler**: Clang with C++20 support (AppleClang on macOS or Clang on Linux)
- **CMake**: Version 3.20 or higher
- **OpenMP**: Required for parallel operations (automatically configured via Homebrew on macOS)
- **Platform**: macOS or Linux

## Building and Testing

### Quick Build Scripts

Debug build and test:
```bash
./debug_build_and_test.sh
```

Release build and test:
```bash
./release_build_and_test.sh
```

### Manual Build

Configure and build (debug):
```bash
cmake --preset clang-debug
cmake --build --preset clang-debug
```

Configure and build (release):
```bash
cmake --preset clang-release
cmake --build --preset clang-release
```

Run tests:
```bash
ctest --preset clang-debug  # or clang-release
```

## Integration

MafLib is designed for easy integration. Since it is header-only, include it in your CMake project:

```cmake
add_subdirectory(path/to/MafLib)
target_link_libraries(your_target PRIVATE MafLib::MafLib)
```

Then include headers as needed:
```cpp
#include <MafLib/math/linalg/Matrix.hpp>
#include <MafLib/math/linalg/Vector.hpp>

int main() {
    maf::math::Matrix<double> m(3, 3);
    // Your code here
    return 0;
}
```

## Documentation

Documentation is generated using Doxygen. The library uses the `maf` namespace with subnamespaces for different components:
- `maf::math` - Mathematical functions and linear algebra
- `maf::util` - Utility functions

Header files contain detailed API documentation.

## Project Status

MafLib is under active development. Current focus areas include:
- Expanding linear algebra operations
- Completing parser implementations
- Improving platform compatibility
- Enhancing performance optimizations
- Adding more comprehensive examples

## Future Plans

- Extended numerical algorithms and special functions
- Additional matrix factorization methods
- More data structure implementations
- Enhanced CSV and data parsing capabilities
- Potential GPU acceleration
- Broader platform support

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Ivan Dzanija
- GitHub: [@IvanDzanija](https://github.com/IvanDzanija)
