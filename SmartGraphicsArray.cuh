#pragma once

#include <SDL.h>
#include <glad.h>
#include <SDL_opengl.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

template<size_t n> //size of component
struct d_gArray {

	size_t size;
	bool mode;
	GLfloat* data;

	d_gArray(): size(0), mode(false), data(nullptr), capacity(0), id(0), attr(0), cgr(nullptr) {
		glGenBuffers(1, &id);
	}

	~d_gArray() {
		cudaGraphicsUnregisterResource(cgr);
		glDeleteBuffers(1, &id);
		cudaFree(data);
	}

	void allocate(size_t c, GLuint a, bool k) {
		capacity = c;
		glBindBuffer(GL_ARRAY_BUFFER, id);
		glBufferData(GL_ARRAY_BUFFER, n * c * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(a, n, GL_FLOAT, k ? 1 : 0, 0, (void*)0);
		glEnableVertexAttribArray(a);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cgr, id, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	void bind(GLuint a, bool k) {
		glBindBuffer(GL_ARRAY_BUFFER, id);
		glVertexAttribPointer(a, n, GL_FLOAT, k ? 1 : 0, 0, (void*)0);
		glEnableVertexAttribArray(a);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void enableCUDA() {
		if (!mode) {
			cudaGraphicsMapResources(1, &cgr, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&data, &size, cgr);
			mode = true;
		}
	}

	void disableCUDA() { if (mode) cudaGraphicsUnmapResources(1, &cgr, 0); mode = false; }

	void h_copy(GLfloat* d, size_t s) {
		if (s <= capacity && mode) {
			enableCUDA();
			cudaMemcpy(data, d, s, cudaMemcpyHostToDevice);
			disableCUDA();
		}
	}

private:

	size_t capacity;
	GLuint id;
	GLuint attr;
	cudaGraphicsResource* cgr;

};
