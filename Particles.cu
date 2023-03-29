#include "Particles.cuh"

// Computes a Dynamic Collision between the exterior of 2 particles.
// Heavily optimized for GPU.
__device__ inline float collideExt(
	float3& p1, // Position of Particle 1
	float3& v1, // Velocity of Particle 1
	const float r1, // Radius of Particle 1
	float3& p2, // Position of Particle 2
	float3& v2, // Velocity of Particle 2
	const float r2, // Radius of Particle 2
	const float cr //Energy Transfer Coeficient. [2.0 for completely elastic][1.0 for completely innelastic]
) {
	const float3 p = p2 - p1, v = v2 - v1; // Relative velocity and position displacement
	const float w = dot(v, v); // Squared length of relative velocity
	const float q = dot(p, v); // Relative position dot relative velocity
	const float k = (w + q); // Used in the boolean expression below
	float r = r1 + r2; r *= r; // Sum the radii squared
	const float o = (dot(p, p) - r); // Ralative position compared with sum of radii squared. Determines if spheres are overlapping. o < 0 -> overlap.
	const float h = w * o; // c in the quadratic equation used to solve for time
	float z = (q * q) - h; // determinate of the quadratic used to solve for time
	//Whether the spheres collide within the given frame or are overlapping:
	const float b = ((z > 0.0f) && ((k > 0.0f) || (z > (k * k))) && (((q < 0.0f) && (h > 0.0f)) || (o < 0.0f))) ? 1.0f : 0.0f;
	const float d = sqrt(z * b); // need this for later when finding the normal dotted with velocity
	z = ((q + d) / w); // solve quadratic to find the time of collision
	// For some reason when I did the math, (v dot c) / (c dot c) = sqrt(z) / (r1 + r2)^2
	// calculate amount of velocity in the direction of the normal
	float3 c = (p - (v * z)) * ((d * cr) / r);
	v1 -= c, v2 += c; // Get new velocity by adding change in velocity.
	c *= z; // Amount of velocity after collision needed to get to the new position.
	p1 += (v1 - c) * b, p2 += (v2 + c) * b; // add amount of velocity after collision to position
	return b; // return that the particles have collided. (Checking for no collision is more likely)
}

//// Computes a Dynamic Collision between 2 particles where one of them is on the interior of the other.
//template<typename T>
//__device__ bool collideInt(
//	dvec<float>& p1,
//	dvec<float>& v1,
//	dvec<float>& p2,
//	dvec<float>& v2,
//	const float r1,
//	const float r2,
//	const float cr
//) {
//	const dvec<float> v = v2 - v1; // Relative Velocity
//	const float w = v.dot(v); //squared distance between velocity vectors
//	const dvec<float> p = p2 - p1; // Relative Position
//	const float q = p.dot(v);
//	const float r = r2 - r1; // dif of radius
//	float z = (q * q) - (w * (p.dot(p) - (r * r))); // Find determinate
//	if(z < 0.0f) return true; // No real solutions means no collision
//	z = (sqrt(z) - q) / w; // Proceed with quadratic formula
//	if(z > 0.0f || z < -1.0f) return true; //Collision is unreliable if it does not happen within current frame.
//	dvec<float> c = p - (v * z); // Relative point of contact.
//	c *= cr * (v.dot(c) / c.dot(c)); // Change in velocity based on normal at point of contact.
//	v1 += c, v2 -= c; // Get new velocity by adding change in velocity
//	c *= z; // Amount of velocity between collision and new position
//	p1 += v1 + c, p2 += v2 - c; // add amount of velocity after collision to position
//	return false;
//}
//
//__device__ bool collideInt(
//	dvec<float>& p1,
//	dvec<float>& v1,
//	const dvec<float> p2,
//	const float r1,
//	const float r2,
//	const float cr
//) {
//	const float w = v1.dot(v1); //squared distance between velocity vectors
//	const dvec<float> p = p2 - p1; // Relative Position
//	const float q = p.dot(v1);
//	const float r = r2 - r1; // sum of radius
//	float z = (q * q) - (w * (p.dot(p) - (r * r))); // Find determinate
//	if(z < 0.0) return false; // No real solutions means no collision
//	z = (q + sqrt(z)) / w; // Proceed with quadratic formula
//	if(z < 0.0 || z > 1.0) return false; //Collision is invalid if it does not happen within current frame.
//	dvec<float> c = p - (v1 * z); // Relative point of contact.
//	c *= cr * (v1.dot(c) / c.dot(c)); // Change in velocity based on normal at point of contact.
//	v1 -= c; // Get new velocity by adding change in velocity
//	p1 += v1 + (c * z); // add amount of velocity after collision to position
//	return true; // return that the particles have collided
//}

__global__ void mortonCodes(float4* pos, float3* vel, size_t* codes, size_t* index, size_t size) {
	size_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if (id < size) {
		float4 p = pos[id];
		float3 v = vel[id];
		uint64_t px = (((uint64_t)((p.x + (0.5 * v.x) + 1048576.0f) * universeCellScale)) & 2097151ull),
				 py = (((uint64_t)((p.y + (0.5 * v.y) + 1048576.0f) * universeCellScale)) & 2097151ull),
				 pz = (((uint64_t)((p.z + (0.5 * v.z) + 1048576.0f) * universeCellScale)) & 2097151ull);

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

		codes[id] = px | (py << 1) | (pz << 2);
		index[id] = id;
	}
}
//
//__global__ void createAABBs(float4* pos, float3* vel, AABB* ab, size_t* index, size_t size) {
//	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
//	if (id < size) {
//		uint32_t sid = index[id];
//		ab[id] = AABB(pos[sid], vel[sid]);
//	}
//}

__global__ void initSpheres(
	float4* pos,
	float4* col,
	float3* vel,
	size_t size
){
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if(id < size){
		float cid = ((float)id) / ((float)size);
		float3 v = float3{0.0f, 0.0f, 0.0f};
		float4 p = float4{0.0f, 0.0f, 0.0f, 60.0f};

		if (id > 0) {
			p = float4{
				((float)((id / 256) % 256) - 128.0f),
				((float)(id % 256) - 128.0f),
				(float)((id / 65536) % 256) + 1024.0f,
				0.5f
			};
			v = float3{
				1.0f,
				0.1f * sin((float)TAU * cid * 1024.0f),
				0.0f
			};
		}

		pos[id] = p;
		vel[id] = v;
		col[id] = float4{	
			(cos((float)TAU * cid)) + 0.5f,
			(cos((float)TAU * (cid + (float)D23))) + 0.5f,
			(cos((float)TAU * (cid + (float)D43))) + 0.5f,
			1.0f
		};
	}
}

__global__ void updateVelocity(
	float4* pos,
	float4* col,
	float3* vel,
	size_t size,
	float3 mvel
){
	uint32_t id = threadIdx.x + (blockDim.x * blockIdx.x);
	if (id < size) {

		//Transfer to local memory
		float4 lp = pos[id];
		float3 p = float3{lp.x, lp.y, lp.z};
		float r = lp.w;
		float3 v = vel[id];

		//Get Mouse Position
		float4 mpl = pos[0];
		float3 mp = float3{mpl.x, mpl.y, mpl.z};

		if (id == 0) {
			v = mvel;
			p += v;
		}
		else {

			//calculate gravity
			const float3 gp = mp - p;
			float d = dot(gp, gp);
			d *= sqrt(d);
			v += (gp / d) * 10000.0f;

			//collide with the big sphere
			p += v * !collideExt(p, v, r, mp, mvel, mpl.w, 2.0f);

		}

		v.x *= 1.0f - ((1.0f + boxres) * ((((p.x + r) > universeSize) && (v.x > 0.0f)) || (((p.x - r) < -universeSize) && (v.x < 0.0f))));
		v.y *= 1.0f - ((1.0f + boxres) * ((((p.y + r) > universeSize) && (v.y > 0.0f)) || (((p.y - r) < -universeSize) && (v.y < 0.0f))));
		v.z *= 1.0f - ((1.0f + boxres) * ((((p.z + r) > universeSize) && (v.z > 0.0f)) || (((p.z - r) < -universeSize) && (v.z < 0.0f))));

		//Transfer back to global memory
		vel[id] = v;
		pos[id] = float4{p.x, p.y, p.z, r};
		//col[id] = float4{v.x, v.y, v.z, 1.0f};

	}
}

ParticleBuffer::ParticleBuffer():
	mousevel(float3{0.0f,0.0f,0.0f}),
	spheres(),
	v(mainPointSize)
	/*index(mainPointSize),
	codes(mainPointSize),
	sup(mainPointSize),
	sub(mainPointSize),
	bounds(mainPointSize)*/
{
	spheres.position.enableCUDA();
	spheres.color.enableCUDA();

	initSpheres<<<blocknum, blockSize>>>(
		(float4*)spheres.position.data,
		(float4*)spheres.color.data,
		v.data,
		mainPointSize
	);

	spheres.position.disableCUDA();
	spheres.color.disableCUDA();
	spheres.enable();

}

ParticleBuffer::~ParticleBuffer(){}

//void ParticleBuffer::createBHV() {
//
//	mortonCodes<<<blocknum, blockSize>>>(
//		(float4*)spheres.position.data,
//		v.data,
//		codes.data,
//		index.data,
//		mainPointSize
//	);
//
//	void* sorttemp = nullptr;
//	size_t sorttempsize = 0;
//	cub::DeviceRadixSort::SortPairs(
//		sorttemp,
//		sorttempsize,
//		codes.data,
//		codes.data,
//		index.data,
//		index.data,
//		mainPointSize
//	);
//
//	cudaMalloc(&sorttemp, sorttempsize);
//
//	cub::DeviceRadixSort::SortPairs(
//		sorttemp,
//		sorttempsize,
//		codes.data,
//		codes.data,
//		index.data,
//		index.data,
//		mainPointSize
//
//	);
//
//	cudaFree(sorttemp);
//
//	createAABBs<<<blocknum, blockSize>>>(
//		(float4*)spheres.position.data,
//		v.data,
//		bounds.data,
//		index.data,
//		mainPointSize
//	);
//
//}

void ParticleBuffer::update(){

	if (Input::lb) mousevel = matrix3{
		float3{Input::cam.iC.x.x, Input::cam.iC.x.y, Input::cam.iC.x.z},
		float3{Input::cam.iC.y.x, Input::cam.iC.y.y, Input::cam.iC.y.z},
		float3{Input::cam.iC.z.x, Input::cam.iC.z.y, Input::cam.iC.z.z}
	} * float3{Input::dx, Input::dy, 0.0f};

	if(Input::key(SDL_SCANCODE_SPACE)){
		spheres.position.enableCUDA();
		spheres.color.enableCUDA();

		if(Input::key(SDL_SCANCODE_Z)){

			initSpheres<<<blocknum, blockSize >>> (
				(float4*)spheres.position.data,
				(float4*)spheres.color.data,
				v.data,
				mainPointSize
			);

		} else {
			//Update the Velocity of the spheres.
			updateVelocity<<<blocknum, blockSize>>>(
				(float4*)spheres.position.data,
				(float4*)spheres.color.data,
				v.data,
				mainPointSize,
				mousevel
			);
		}
		spheres.position.disableCUDA();
		spheres.color.disableCUDA();
	}
	spheres.draw(mainPointSize);
}