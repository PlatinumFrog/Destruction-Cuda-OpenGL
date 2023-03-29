#pragma once
#include <chrono>
#include "SmartDeviceArray.cuh"
#include "VAOnD.cuh"
#include "Constants.cuh"
#include "AABB.cuh"
//#include "BVH.cuh"

constexpr float universeSize = 4096.0f;
constexpr float quantizeDepth = 1048576.0f;
constexpr float universeCellScale = quantizeDepth / universeSize;
constexpr float boxres = 0.8f;


constexpr size_t blockSize = 256;
constexpr size_t blocknum = (mainPointSize / blockSize) + 1ull;

class ParticleBuffer{
	float3 mousevel;
	VAOSphere<mainPointSize> spheres;

	d_Array<float3> v;

	/*d_Array<uint64_t> index;
	d_Array<uint64_t> codes;
	
	d_Array<AABB> bounds;

	d_Array<uint64_t> sup;
	d_Array<uint64_t> sub;*/

	/*void createBHV();*/

public:

	ParticleBuffer();
	~ParticleBuffer();

	void update();

};