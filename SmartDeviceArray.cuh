#pragma once
//#include <string>
//#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matrix.cuh"
#include "AABB.cuh"

template<typename T>
struct d_Array {

	size_t size, capacity;
	T *data;

	d_Array();
	d_Array(size_t s);
	~d_Array();

	void allocate(size_t s);
	void h_insert(T *d, size_t s);
	void d_insert(T *d, size_t s);
	void print();

};

