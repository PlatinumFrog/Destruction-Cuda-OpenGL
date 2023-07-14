#pragma once
#include <chrono>
#include "SmartDeviceArray.cuh"
#include "VAOnD.cuh"
#include "Constants.cuh"
#include "BVH.cuh"

class ParticleBuffer{
	float3 mousevel;
	VAOSphere<mainPointSize> spheres;

	d_Array<float3> v;
	BVH b;
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