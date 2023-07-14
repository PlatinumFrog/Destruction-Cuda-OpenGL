#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "SmartDeviceArray.cuh"
#include "AABB.cuh"
#include "vector.cuh"
#include "Constants.cuh"
//#define USING_64_BIT_INDEXES

//struct intNode2 {
//	AABB aabb;
//	uint32_t left, super;
//};

//struct intNode {
//	uint32_t flags;
//	float3 pos;
//	uint64_t left, right, super;
//	AABB aabb;
//}; // 64 bytes
//
//struct leafNode {
//	uint64_t super;
//	AABB aabb;
//}; // 32 bytes

struct BVH {

	AABB * iAABB, * lAABB;
	uint64_t* codes;
	
#ifdef USING_64_BIT_INDEXES
	uint64_t* iSuper, * iLeft, * iRight, * lSuper, * indexes;
	uint64_t size, nBlocks, nThreads;
#else
	uint32_t* iSuper, * iLeft, * iRight, * lSuper, * indexes;
	uint32_t size, nBlocks, nThreads;
#endif
	uint32_t * tempInc, * iFlags;
	
	BVH():
		iAABB(nullptr),
		lAABB(nullptr),
		codes(nullptr),
		iSuper(nullptr),
		iLeft(nullptr),
		iRight(nullptr),
		lSuper(nullptr),
		indexes(nullptr),
		size(0u),
		nBlocks(0u),
		nThreads(0u),
		tempInc(nullptr),
		iFlags(nullptr)
	{}

	BVH(size_t s):
		iAABB(nullptr),
		lAABB(nullptr),
		codes(nullptr),
		iSuper(nullptr),
		iLeft(nullptr),
		iRight(nullptr),
		lSuper(nullptr),
		indexes(nullptr),
		size(s),
		nBlocks(s),
		nThreads(s),
		tempInc(nullptr),
		iFlags(nullptr)
	{

		cudaMalloc((void**)&iAABB, (size - 1u) * sizeof(AABB));
		cudaMalloc((void**)&lAABB, size * sizeof(AABB));
		cudaMalloc((void**)&codes, size * sizeof(uint64_t));
#ifdef USING_64_BIT_INDEXES
		cudaMalloc((void**)&indexes, size * sizeof(uint64_t));
		cudaMalloc((void**)&iLeft, (size - 1u) * sizeof(uint64_t));
		cudaMalloc((void**)&iRight, (size - 1u) * sizeof(uint64_t));
		cudaMalloc((void**)&iSuper, (size - 1u) * sizeof(uint64_t));
		cudaMalloc((void**)&lSuper, size * sizeof(uint64_t));
#else
		cudaMalloc((void**)&indexes, size * sizeof(uint32_t));
		cudaMalloc((void**)&iLeft, (size - 1u) * sizeof(uint32_t));
		cudaMalloc((void**)&iRight, (size - 1u) * sizeof(uint32_t));
		cudaMalloc((void**)&iSuper, (size - 1u) * sizeof(uint32_t));
		cudaMalloc((void**)&lSuper, size * sizeof(uint32_t));
#endif
		cudaMalloc((void**)&iFlags, (size - 1u) * sizeof(uint32_t));
		cudaMalloc((void**)&tempInc, size * sizeof(uint32_t));

		nThreads--;
		nThreads |= nThreads >> 1u;
		nThreads |= nThreads >> 2u;
		nThreads |= nThreads >> 4u;
		nThreads |= nThreads >> 8u;
		nThreads |= nThreads >> 16u;
#ifdef USING_64_BIT_INDEXES
		nThreads |= nThreads >> 32u;
#endif
		nThreads++;
		nThreads = (nThreads < blockSize) ? nThreads : blockSize;
		nBlocks = ((size - 1ull) / nThreads) + 1ull;
	}

	~BVH() {
		cudaFree(iAABB);
		cudaFree(lAABB);
		cudaFree(codes);
		cudaFree(iSuper);
		cudaFree(iLeft);
		cudaFree(iRight);
		cudaFree(lSuper);
		cudaFree(indexes);
		cudaFree(tempInc);
		cudaFree(iFlags);
	}

	void build(float4* pos, float3* vel);
	
	void build(float4* pos, float3* vel, size_t s);

	void printCodes();

	void printNodes();

};






//#pragma once
//#include <chrono>
//
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//
//#include <thrust/execution_policy.h>
//#include <thrust/sort.h>
//
//
//
//#include "constants.cuh"
//#include "vector.cuh"
//#include "AABB.cuh"
//
//#define BUILDBVH
//
//struct BVH {
//
//	AABB 
//		*boxInt,
//		*boxLeaf;
//
//	float3* nodePos;
//
//	uint32_t
//		*indexes,
//
//		*superInt, 
//		*superLeaf, 
//
//		*left, 
//		*right,
//
//		*tempBox;
//
//	uint64_t* codes;
//		
//
//	bool 
//		*lLeaf, 
//		*rLeaf;
//
//	uint64_t size;
//
//	BVH():
//		boxInt(nullptr),
//		boxLeaf(nullptr),
//
//		tempBox(nullptr),
//		nodePos(nullptr),
//
//		codes(nullptr),
//		indexes(nullptr),
//
//		superInt(nullptr),
//		superLeaf(nullptr),
//
//		left(nullptr),
//		right(nullptr),
//
//		lLeaf(nullptr),
//		rLeaf(nullptr),
//
//		size(0)
//	{}
//
//	BVH(size_t s):
//		boxInt(nullptr),
//		boxLeaf(nullptr),
//
//		tempBox(nullptr),
//		nodePos(nullptr),
//
//		codes(nullptr),
//		indexes(nullptr),
//
//		superInt(nullptr),
//		superLeaf(nullptr),
//
//		left(nullptr),
//		right(nullptr),
//
//		lLeaf(nullptr),
//		rLeaf(nullptr),
//
//		size(s)
//	{
//		cudaMalloc((void**)&boxInt, (size - 1ull) * sizeof(AABB));
//		cudaMalloc((void**)&boxLeaf, size * sizeof(AABB));
//
//		cudaMalloc((void**)&tempBox, (size - 1ull) * sizeof(unsigned int));
//		cudaMalloc((void**)&nodePos, (size - 1ull) * sizeof(float3));
//
//		cudaMalloc((void**)&codes, size * sizeof(uint64_t));
//		cudaMalloc((void**)&indexes, size * sizeof(uint64_t));
//
//		cudaMalloc((void**)&superInt, (size - 1ull) * sizeof(uint64_t));
//		cudaMalloc((void**)&superLeaf, (size - 1ull) * sizeof(uint64_t));
//
//		cudaMalloc((void**)&left, (size - 1ull) * sizeof(uint64_t));
//		cudaMalloc((void**)&right, (size - 1ull) * sizeof(uint64_t));
//
//		cudaMalloc((void**)&lLeaf, (size - 1ull) * sizeof(bool));
//		cudaMalloc((void**)&rLeaf, (size - 1ull) * sizeof(bool));
//	}
//
//	~BVH() {
//		cudaFree(boxInt);
//		cudaFree(boxLeaf);
//
//		cudaFree(tempBox);
//		cudaFree(nodePos);
//
//		cudaFree(codes);
//		cudaFree(indexes);
//
//		cudaFree(superInt);
//		cudaFree(superLeaf);
//
//		cudaFree(left);
//		cudaFree(right);
//
//		cudaFree(lLeaf);
//		cudaFree(rLeaf);
//	}
//
//	double build(float4* pos, float3* vel);
//	double build(float4* pos, float3* vel, size_t s);
//};


