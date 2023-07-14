#pragma once
#include "vector.cuh"

struct AABB {
	float3 upper, lower;

	__host__ __device__ AABB(): 
		upper({0.0f, 0.0f, 0.0f}),
		lower({0.0f, 0.0f, 0.0f})
	{}

	__host__ __device__ AABB(float3 upper, float3 lower): upper(upper), lower(lower) {};

	__host__ __device__ AABB(float3 p, float3 v, float r) {
		float3 d = p + v;
		bool bx = v.x > 0.0f, by = v.y > 0.0f, bz = v.z > 0.0f;
		r = abs(r);
		upper = float3{
			(bx ? d.x : p.x) + r,
			(by ? d.y : p.y) + r,
			(bz ? d.z : p.z) + r,
		};
		lower = float3{
			(bx ? p.x : d.x) - r,
			(by ? p.y : d.y) - r,
			(bz ? p.z : d.z) - r,
		};
	};

	__host__ __device__ AABB(float4 p, float3 v) {
		p.w = abs(p.w);
		float3 d = float3{p.x, p.y, p.z} + v;
		bool bx = v.x > 0.0f, by = v.y > 0.0f, bz = v.z > 0.0f;
		upper = float3{
			(bx ? d.x : p.x) + p.w,
			(by ? d.y : p.y) + p.w,
			(bz ? d.z : p.z) + p.w,
		};
		lower = float3{
			(bx ? p.x : d.x) - p.w,
			(by ? p.y : d.y) - p.w,
			(bz ? p.z : d.z) - p.w,
		};
	};

	__host__ __device__ inline AABB operator+(AABB aabb) {
		return AABB{
			float3{
				(upper.x > aabb.upper.x) ? upper.x : aabb.upper.x,
				(upper.y > aabb.upper.y) ? upper.y : aabb.upper.y,
				(upper.z > aabb.upper.z) ? upper.z : aabb.upper.z
			},
			float3{
				(lower.x < aabb.lower.x) ? lower.x : aabb.lower.x,
				(lower.y < aabb.lower.y) ? lower.y : aabb.lower.y,
				(lower.z < aabb.lower.z) ? lower.z : aabb.lower.z
			}
		};
	};

	__host__ __device__ inline AABB operator+=(AABB aabb) {
		return AABB{
			upper = float3{
				(upper.x > aabb.upper.x) ? upper.x : aabb.upper.x,
				(upper.y > aabb.upper.y) ? upper.y : aabb.upper.y,
				(upper.z > aabb.upper.z) ? upper.z : aabb.upper.z
			},
			lower = float3{
				(lower.x < aabb.lower.x) ? lower.x : aabb.lower.x,
				(lower.y < aabb.lower.y) ? lower.y : aabb.lower.y,
				(lower.z < aabb.lower.z) ? lower.z : aabb.lower.z
			}
		};
	};

	// * will be used to test intersection between AABBs because I don't want to write 'intersection' every time I need an intersection test.

	//Needs testing to see which is faster but for now I am using this one.
	__host__ __device__ inline bool operator*(AABB aabb) {
		return
			((((upper.x > aabb.upper.x) ? upper.x : aabb.upper.x) - ((lower.x < aabb.lower.x) ? lower.x : aabb.lower.x)) < ((upper.x - lower.x) + (aabb.upper.x - aabb.lower.x))) ||
			((((upper.y > aabb.upper.y) ? upper.y : aabb.upper.y) - ((lower.y < aabb.lower.y) ? lower.y : aabb.lower.y)) < ((upper.y - lower.y) + (aabb.upper.y - aabb.lower.y))) ||
			((((upper.z > aabb.upper.z) ? upper.z : aabb.upper.z) - ((lower.z < aabb.lower.z) ? lower.z : aabb.lower.z)) < ((upper.z - lower.z) + (aabb.upper.z - aabb.lower.z)));
	}

	__host__ __device__ inline float3 center() { return (upper + lower) * 0.5; };
};
/*__host__ __device__ inline bool operator*(AABB aabb) {
		return
			((lower.x > aabb.lower.x) && (lower.x < aabb.upper.x)) ||
			((upper.x > aabb.lower.x) && (upper.x < aabb.upper.x)) ||
			((aabb.lower.x > lower.x) && (aabb.lower.x < upper.x)) ||
			((aabb.upper.x > lower.x) && (aabb.upper.x < upper.x)) ||

			((lower.y > aabb.lower.y) && (lower.y < aabb.upper.y)) ||
			((upper.y > aabb.lower.y) && (upper.y < aabb.upper.y)) ||
			((aabb.lower.y > lower.y) && (aabb.lower.y < upper.y)) ||
			((aabb.upper.y > lower.y) && (aabb.upper.y < upper.y)) ||

			((lower.z > aabb.lower.z) && (lower.z < aabb.upper.z)) ||
			((upper.z > aabb.lower.z) && (upper.z < aabb.upper.z)) ||
			((aabb.lower.z > lower.z) && (aabb.lower.z < upper.z)) ||
			((aabb.upper.z > lower.z) && (aabb.upper.z < upper.z));
	};*/
//struct VABB {
//	float3 bounds[8];
//	VABB(const float3 pos, const float3 vel, const float r) {
//		const float3 forward = r * norm(vel);
//		const float3 right = r * norm(float3{-vel.z, 0.0, vel.x});
//		const float3 up = r * norm(float3{-vel.x * vel.y, (vel.z * vel.z) + (vel.x * vel.x), -vel.y * vel.z});
//		bounds[0] = up + right;
//		bounds[2] = pos - bounds[0];
//		bounds[0] += pos;
//		bounds[1] = up - right;
//		bounds[3] = pos - bounds[1];
//		bounds[1] += pos;
//		const float3 forvel = forward + vel;
//		bounds[4] = forvel + bounds[0];
//		bounds[0] -= forward;
//		bounds[5] = forvel + bounds[1];
//		bounds[1] -= forward;
//		bounds[6] = forvel + bounds[2];
//		bounds[2] -= forward;
//		bounds[7] = forvel + bounds[3];
//		bounds[3] -= forward;
//	}
//};