#include "SmartDeviceArray.cuh"

template<typename T>
__global__ void printDeviceArray(T *data, size_t s) {}

template<>
__global__ void printDeviceArray(uint32_t *data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if(id < s) printf("[%d:(%d, %d)]: %d\n", id, blockIdx.x, threadIdx.x, data[id]);
}

template<>
__global__ void printDeviceArray(size_t *data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if(id < s) printf("[%d:(%d, %d)]: %zu\n", id, blockIdx.x, threadIdx.x, data[id]);
}

template<>
__global__ void printDeviceArray(float *data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if(id < s) printf("[%d:(%d, %d)]: %f\n", id, blockIdx.x, threadIdx.x, data[id]);
}

template<>
__global__ void printDeviceArray(double *data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if(id < s) printf("[%d:(%d, %d)]: %f\n", id, blockIdx.x, threadIdx.x, data[id]);
}

template<>
__global__ void printDeviceArray(float2* data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	float2 d = data[id];
	if (id < s) printf("[%d:(%d, %d)]: {%f, %f}\n", id, blockIdx.x, threadIdx.x, d.x, d.y);
}

template<>
__global__ void printDeviceArray(float3* data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	float3 d = data[id];
	if (id < s) printf("[%d:(%d, %d)]: {%f, %f, %f}\n", id, blockIdx.x, threadIdx.x, d.x, d.y, d.z);
}

template<>
__global__ void printDeviceArray(float4* data, size_t s) {
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	float4 d = data[id];
	if (id < s) printf("[%d:(%d, %d)]: {%f, %f, %f, %f}\n", id, blockIdx.x, threadIdx.x, d.x, d.y, d.z, d.w);
}

//template <unsigned int blockSize>
//__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
//	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//}
//
//template <unsigned int blockSize>
//__global__ void sumReduce(int* g_idata, int* g_odata, unsigned int n) {
//	extern __shared__ int sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata[tid] = 0;
//	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) warpReduce(sdata, tid);
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}



template<typename T>
d_Array<T>::d_Array(): size(0), capacity(0), data(nullptr) {}

template<typename T>
d_Array<T>::d_Array(size_t s): size(0), capacity(s), data(nullptr) {
	cudaMalloc((void **)&data, s * sizeof(T));
}

template<typename T>
d_Array<T>::~d_Array() { cudaFree(data); }

template<typename T>
void d_Array<T>::allocate(size_t s) {
	if(data != nullptr) cudaFree(data);
	cudaMalloc((void **)&data, s * sizeof(T)); capacity = s;
}

template<typename T>
void d_Array<T>::h_insert(T *d, size_t s) {
	if(size < capacity && capacity > 0 && data != nullptr) {
		if(s + size > capacity) s = capacity - size;
		cudaMemcpy(data + size, d, s * sizeof(T), cudaMemcpyHostToDevice);
		size += s;
	}
}

template<typename T>
void d_Array<T>::d_insert(T *d, size_t s) {
	if(size < capacity && capacity > 0 && data != nullptr) {
		if(s + size > capacity) s = capacity - size;
		cudaMemcpy(data + size, d, s * sizeof(T), cudaMemcpyDeviceToDevice);
		size += s;
	}
}

template<typename T>
void d_Array<T>::print() {
	printDeviceArray<<<(int)((size / size_t(32)) + size_t(1)), 32>>>(data, size);
}

template class d_Array<uint32_t>;
template class d_Array<size_t>;
template class d_Array<float>;
template class d_Array<double>;
template class d_Array<float2>;
template class d_Array<float3>;
template class d_Array<float4>;
template class d_Array<AABB>;