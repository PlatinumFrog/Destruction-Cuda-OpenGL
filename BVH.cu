#include "BVH.cuh"

__global__ void printinode(
	AABB* iAABB,
#ifdef USING_64_BIT_INDEXES
	uint64_t size,
	uint64_t* iLeft,
	uint64_t* iRight,
	uint64_t* iSuper,
#else
	uint32_t size,
	uint32_t* iLeft,
	uint32_t* iRight,
	uint32_t* iSuper,
#endif
	uint32_t* iFlags
) {
#ifdef USING_64_BIT_INDEXES
	uint64_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#else
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#endif
	if (id < (size - 1)) {
		uint32_t flags = iFlags[id];
		//AABB A = iAABB[id];
#ifdef USING_64_BIT_INDEXES
		printf(
			"[%2llu <- %2llu -> %2llu%-1c|%2llu%-1c]: Upper: <%f, %f, %f>, Lower: <%f, %f, %f>\n",
			iSuper[id],
			id,
			iLeft[id],
			((flags & 1) ? 'L' : ' '), 
			iRight[id],
			((flags & 2) ? 'L' : ' '),
			A.upper.x,
			A.upper.y,
			A.upper.z,
			A.lower.x,
			A.lower.y,
			A.lower.z
		);: Upper: <%f, %f, %f>, Lower: <%f, %f, %f>
#else
		printf(
			"[%2lu <- %2lu -> %2lu%-1c|%2lu%-1c]\n",
			iSuper[id],
			id,
			iLeft[id],
			((flags & 1) ? 'L' : ' '),
			iRight[id],
			((flags & 2) ? 'L' : ' ')/*,
			A.upper.x,
			A.upper.y,
			A.upper.z,
			A.lower.x,
			A.lower.y,
			A.lower.z*/
		);
#endif
	}
}

__global__ void mortonCodes(
	float4* pos,
	uint64_t* codes,
#ifdef USING_64_BIT_INDEXES
	uint64_t* indexes,
	uint64_t size
#else
	uint32_t* indexes,
	uint32_t size
#endif
	
) {
#ifdef USING_64_BIT_INDEXES
	uint64_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#else
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#endif
	if (id < size) {
		float4 p = pos[id];
		p.w = abs(p.w * 0.3f);
		pos[id] = p;

		uint64_t
			px = (((uint64_t)((p.x * universeScale) + 1048576.0f)) & 2097151ull),
			py = (((uint64_t)((p.y * universeScale) + 1048576.0f)) & 2097151ull),
			pz = (((uint64_t)((p.z * universeScale) + 1048576.0f)) & 2097151ull);

		px = (px | px << 32) & 0x1f00000000ffff;
		px = (px | px << 16) & 0x1f0000ff0000ff;
		px = (px | px << 8) & 0x100f00f00f00f00f;
		px = (px | px << 4) & 0x10c30c30c30c30c3;
		px = (px | px << 2) & 0x1249249249249249;

		py = (py | py << 32) & 0x1f00000000ffff;
		py = (py | py << 16) & 0x1f0000ff0000ff;
		py = (py | py << 8) & 0x100f00f00f00f00f;
		py = (py | py << 4) & 0x10c30c30c30c30c3;
		py = (py | py << 2) & 0x1249249249249249;

		pz = (pz | pz << 32) & 0x1f00000000ffff;
		pz = (pz | pz << 16) & 0x1f0000ff0000ff;
		pz = (pz | pz << 8) & 0x100f00f00f00f00f;
		pz = (pz | pz << 4) & 0x10c30c30c30c30c3;
		pz = (pz | pz << 2) & 0x1249249249249249;

		codes[id] = (px | (py << 1) | (pz << 2)) & bitMask;

		indexes[id] = id;
	}
}
#ifdef USING_64_BIT_INDEXES
__device__ inline int64_t countDif(uint64_t size, uint64_t* codes, uint64_t ic, int64_t i, int64_t j) {
	if ((j < 0ll) || (j >= size)) return -1ll;
	else {
		uint64_t jCode = codes[j];
		return (ic == jCode) ? __clzll(i ^ j) + bitDepth : __clzll(ic ^ jCode);
	}
}
#else
__device__ inline int32_t countDif(uint32_t size, uint64_t* codes, uint64_t ic, int32_t i, int32_t j) {
	if ((j < 0) || (j >= size)) return -1;
	else {
		uint64_t jCode = codes[j];
		return (ic == jCode) ? __clz(i ^ j) + bitDepth: __clzll(ic ^ jCode);
	}
}
#endif
__global__ void buildBVH(
	uint64_t* codes,
#ifdef USING_64_BIT_INDEXES
	uint64_t size,
	uint64_t* indexes,
	uint64_t* iLeft,
	uint64_t* iRight,
	uint64_t* iSuper,
	uint64_t* lSuper,
#else
	uint32_t size,
	uint32_t* indexes,
	uint32_t* iLeft,
	uint32_t* iRight,
	uint32_t* iSuper,
	uint32_t* lSuper,
#endif
	uint32_t* iFlags,
	uint32_t* tempInc
) {
#ifdef USING_64_BIT_INDEXES
	uint64_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if (id < (size - 1ull)) {
		uint64_t co = codes[id];
		int64_t leadl = countDif(size, codes, co, id, id - 1ull), leadr = countDif(size, codes, co, id, id + 1ull);
		bool b = (leadr < leadl);
		int64_t d = (b ? -1ll : 1ll), difmin = b ? leadr : leadl, difmax = 2ll;
		while (countDif(size, codes, co, id, id + (difmax * d)) > difmin) difmax <<= 1ull;
		uint64_t l = 0ull;
		for (uint64_t t = difmax >> 1ull; t > 0ull; t >>= 1ull) if (countDif(size, codes, co, id, id + ((l + t) * d)) > difmin) l += t;
		uint64_t j = id + (l * d), s = 0ull;
		int64_t difnode = countDif(size, codes, co, id, j);
		for (l = (l + 1ull) >> 1ull; l > 0ull; l = (l == 1ull) ? 0ull : (l + 1ull) >> 1ull) if (countDif(size, codes, co, id, id + ((s + l) * d)) > difnode) s += l;
		uint64_t splitl = id + (s * d) + llmin(d, 0ll);
		uint64_t splitr = splitl + 1ull;
		bool ls = (ullmin(id, j) == splitl);
		bool rs = (ullmax(id, j) == splitr);
		iFlags[id] = (ls ? 1u : 0u) | (rs ? 2u : 0u);
		uint64_t indexLeft = (ls ? indexes[splitl] : splitl);
		uint64_t indexRight = (rs ? indexes[splitr] : splitr);
		iLeft[id] = indexLeft;
		iRight[id] = indexRight;
		(ls ? lSuper : iSuper)[indexLeft] = id;
		(rs ? lSuper : iSuper)[indexRight] = id;
		tempInc[id] = 0u;
	}
	if (id == 0ull) iSuper[0ull] = 0ull;
#else
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if (id < (size - 1u)) {
		uint64_t co = codes[id];
		int32_t leadl = countDif(size, codes, co, id, id - 1u), leadr = countDif(size, codes, co, id, id + 1ull);
		bool b = (leadr < leadl);
		int32_t d = (b ? -1 : 1), difmin = b ? leadr : leadl, difmax = 2;
		while (countDif(size, codes, co, id, id + (difmax * d)) > difmin) difmax <<= 1u;
		uint32_t l = 0u;
		for (uint32_t t = difmax >> 1u; t > 0u; t >>= 1u) if (countDif(size, codes, co, id, id + ((l + t) * d)) > difmin) l += t;
		uint32_t j = id + (l * d), s = 0u;
		int32_t difnode = countDif(size, codes, co, id, j);
		for (l = (l + 1u) >> 1u; l > 0u; l = (l == 1u) ? 0u : (l + 1u) >> 1u) if (countDif(size, codes, co, id, id + ((s + l) * d)) > difnode) s += l;
		uint32_t splitl = id + (s * d) + min(d, 0);
		uint32_t splitr = splitl + 1u;
		bool ls = (umin(id, j) == splitl);
		bool rs = (umax(id, j) == splitr);
		iFlags[id] = (ls ? 1u : 0u) | (rs ? 2u : 0u);
		uint32_t indexLeft = (ls ? indexes[splitl] : splitl);
		uint32_t indexRight = (rs ? indexes[splitr] : splitr);
		iLeft[id] = indexLeft;
		iRight[id] = indexRight;
		(ls ? lSuper : iSuper)[indexLeft] = id;
		(rs ? lSuper : iSuper)[indexRight] = id;
		tempInc[id] = 0u;
	}
	if (id == 0u) iSuper[0u] = 0u;
#endif
}

__global__ void constructAABBs(

	AABB* iAABB,
	AABB* lAABB,

	float4* pos,
	float3* vel,

#ifdef USING_64_BIT_INDEXES
	uint64_t size,
	uint64_t* indexes,
	uint64_t* iLeft,
	uint64_t* iRight,
	uint64_t* iSuper,
	uint64_t* lSuper,
#else
	uint32_t size,
	uint32_t* indexes,
	uint32_t* iLeft,
	uint32_t* iRight,
	uint32_t* iSuper,
	uint32_t* lSuper,
#endif

	uint32_t* iFlags,
	uint32_t* tempInc
) {
#ifdef USING_64_BIT_INDEXES
	uint64_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#else
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
#endif
	if (id < size) {
#ifdef USING_64_BIT_INDEXES
		uint64_t k = lSuper[indexes[id]];
#else
		uint32_t k = lSuper[indexes[id]];
#endif
		while (atomicInc(&tempInc[k], 1u) == 1u) {
			
#ifdef USING_64_BIT_INDEXES
			uint64_t left = iLeft[k], right = iRight[k];
#else
			uint32_t left = iLeft[k], right = iRight[k];
#endif
			AABB A;
			uint32_t flags = iFlags[k];
			if (flags & 1u) A = (lAABB[left] = AABB(pos[left], vel[left]));
			else A = iAABB[left];

			if (flags & 2u) A += (lAABB[right] = AABB(pos[right], vel[right]));
			else A += iAABB[right];

			iAABB[k] = A;

#ifdef USING_64_BIT_INDEXES
			uint64_t super = iSuper[k];
#else
			uint32_t super = iSuper[k];
#endif

			if (k == super) break;
			k = super;
		}
	}
}

void BVH::build(float4* pos, float3* vel)
{
	mortonCodes<<<nBlocks, nThreads>>>(pos, codes, indexes, size);
#ifdef _DEBUG
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	thrust::stable_sort_by_key(thrust::device, codes, codes + size, indexes);
#ifdef _DEBUG
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	buildBVH<<<nBlocks, nThreads>>>(
		codes,
		size,
		indexes,
		iLeft,
		iRight,
		iSuper,
		lSuper,
		iFlags,
		tempInc
	);
#ifdef _DEBUG
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	printNodes();
#endif
	constructAABBs<<<nBlocks, nThreads>>>(
		iAABB,
		lAABB,
		pos,
		vel,
		size,
		indexes,
		iLeft,
		iRight,
		iSuper,
		lSuper,
		iFlags,
		tempInc
	);
#ifdef _DEBUG
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	std::cout << "\n\n";
	printNodes();
#endif
}


void BVH::build(float4* pos, float3* vel, size_t s)
{
	if (s != size) {
		size = s;
		nThreads = size;
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
		nThreads = (nThreads < maxThreadsPerBlock) ? nThreads : maxThreadsPerBlock;
		nBlocks = ((size - 1u) / nThreads) + 1u;

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

		cudaMalloc((void**)&iAABB, (size - 1) * sizeof(AABB));
		cudaMalloc((void**)&lAABB, size * sizeof(AABB));
		cudaMalloc((void**)&codes, size * sizeof(uint64_t));
#ifdef USING_64_BIT_INDEXES
		cudaMalloc((void**)&indexes, size * sizeof(uint64_t));
		cudaMalloc((void**)&iLeft, (size - 1) * sizeof(uint64_t));
		cudaMalloc((void**)&iRight, (size - 1) * sizeof(uint64_t));
		cudaMalloc((void**)&iSuper, (size - 1) * sizeof(uint64_t));
		cudaMalloc((void**)&lSuper, size * sizeof(uint64_t));
#else
		cudaMalloc((void**)&indexes, size * sizeof(uint32_t));
		cudaMalloc((void**)&iLeft, (size - 1) * sizeof(uint32_t));
		cudaMalloc((void**)&iRight, (size - 1) * sizeof(uint32_t));
		cudaMalloc((void**)&iSuper, (size - 1) * sizeof(uint32_t));
		cudaMalloc((void**)&lSuper, size * sizeof(uint32_t));
#endif
		cudaMalloc((void**)&iFlags, (size - 1) * sizeof(uint32_t));
		cudaMalloc((void**)&tempInc, size * sizeof(uint32_t));

	}
	mortonCodes<<<nBlocks, nThreads>>>(pos, codes, indexes, size);
	thrust::sort_by_key(thrust::device, codes, codes + size, indexes);
	buildBVH<<<nBlocks, nThreads>>>(
		codes,
		size,
		indexes,
		iLeft,
		iRight,
		iSuper,
		lSuper,
		iFlags,
		tempInc
	);
#ifdef _DEBUG
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	constructAABBs<<<nBlocks, nThreads>>>(
		iAABB,
		lAABB,
		pos,
		vel,
		size,
		indexes,
		iLeft,
		iRight,
		iSuper,
		lSuper,
		iFlags,
		tempInc
	);
}

void BVH::printCodes()
{

}

void BVH::printNodes()
{
	printinode<<<nBlocks, nThreads>>>(
		iAABB,
		size,
		iLeft,
		iRight,
		iSuper,
		iFlags
	);


}

//template class d_Array<intNode>;
//template class d_Array<leafNode>;
//template<>
//void d_Array<intNode>::h_get(intNode* input, intNode* output, uint64_t size)
//{
//	cudaMemcpy(output, input, size * sizeof(intNode), cudaMemcpyDeviceToHost);
//}
//template<>
//void d_Array<leafNode>::h_get(leafNode* input, leafNode* output, uint64_t size)
//{
//	cudaMemcpy(output, input, size * sizeof(leafNode), cudaMemcpyDeviceToHost);
//}
//{
//	sorted = true;
//	for (uint64_t i = 0ull; (i < (nParticles - 1)) && sorted; i++) if (code[i] > code[i + 1]) sorted = false;
//	for (uint64_t i = 1ull; (i < nParticles) && sorted; i++) if (code[i - 1] > code[i]) sorted = false;
//	std::cout << (sorted ? "SORTED!!!" : "NOT SORTED :(") << '\n';
//
//	std::cout << "   |";
//	for (uint64_t i = 0ull; i < nParticles; i++) std::cout << std::setw(2) << i << '|';
//	std::cout << "\n";
//	for (uint64_t j = bitDepth; j > 0; j--) {
//		if (j < 10ull) std::cout << "|0" << j << '|';
//		else std::cout << '|' << j << '|';
//		for (uint64_t i = 0ull; i < nParticles; i++) std::cout << std::setw(2) << ((code[i] & (1ull << (j - 1ull))) ? '1' : '0') << '|';
//		std::cout << '\n';
//	}
//	std::cout << "   |";
//	for (uint64_t i = 0ull; i < nParticles; i++) std::cout << std::setw(2) << index[i] << '|';
//	std::cout << "\n\n|";
//	for (uint64_t i = 0ull; i < nParticles; i++) std::cout << std::setw(2) << code[i] << '|';
//	std::cout << "\n\n\n";
//}
