#pragma once
#include <iostream>
#include <cuda_runtime.h>

constexpr double PI = 3.141592653589793;  // pi
constexpr double TAU = 6.283185307179586; // 2pi
constexpr double RT2 = 1.4142135623730951; // sqrt(2)
constexpr double RT3 = 1.7320508075688773; // sqrt(3)
constexpr double H2B = 1.1547005383792517; // triangle height to base conversion factor: 2/sqrt(3)
constexpr double B2H = 0.8660254037844386; // triangle base to height conversion factor: sqrt(3)/2
constexpr double D13 = 0.3333333333333333; // 1/3
constexpr double D23 = 0.6666666666666667; // 2/3
constexpr double D43 = 1.3333333333333333; // 4/3
constexpr double D16 = 0.1666666666666667; // 1/6
constexpr double DR3 = 0.5773502691896258; // 1/sqrt(3)

//constexpr float PI = 3.1415927;  // pi
//constexpr float TAU = 6.2831853; // 2pi
//constexpr float RT2 = 1.4142135; // sqrt(2)
//constexpr float RT3 = 1.7320508; // sqrt(3)
//constexpr float H2B = 1.1547005; // triangle height to base conversion factor: 2/sqrt(3)
//constexpr float B2H = 0.8660254; // triangle base to height conversion factor: sqrt(3)/2
//constexpr float D13 = 0.3333333; // 1/3
//constexpr float D23 = 0.6666667; // 2/3
//constexpr float D43 = 1.3333333; // 4/3
//constexpr float D16 = 0.1666667; // 1/6
//constexpr float DR3 = 0.5773503; // 1/sqrt(3)

//constexpr size_t mainPointSize = 32;
//constexpr size_t mainPointSize = 2048;
//constexpr size_t mainPointSize = 65536;
//constexpr size_t mainPointSize = 1000000;
//constexpr size_t mainPointSize = 2073600;
constexpr size_t mainPointSize = 2097152;
//constexpr size_t mainPointSize = 4194304;
//constexpr size_t mainPointSize = 8388608;
//constexpr size_t mainPointSize = 16777216;

constexpr uint64_t blockSize = 256ull;
constexpr size_t blockNum = (mainPointSize / blockSize);
constexpr uint64_t bitDepth = 63ull;
constexpr uint64_t bitMask = (1ull << bitDepth) - 1ull;
constexpr float universeSize = 1024.0f;
constexpr float universeScale = 1048576.0f / universeSize;
constexpr float boxres = 0.8f;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}