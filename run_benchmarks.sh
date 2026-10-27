cmake --preset clang-release
cmake --build --preset clang-release --target LinAlgMeasure
mkdir -p measure/results
./build/clang-release/measure/LinAlgMeasure | tee measure/results/linalg_$(date +%Y%m%d_%H%M%S).txt
