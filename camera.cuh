#pragma once

#include "Constants.cuh"
#include "matrix.cuh"

constexpr float near = 0.0000001f;

constexpr float far = 10000.0f;

constexpr float pm1 = ((far + near) / (near - far));

constexpr float pm2 = ((2.0f * near * far) / (near - far));

constexpr float asp = 1080.0f/1920.0f;

struct camera {

	float fv;

	float2 rot;

	float3 pos;
	
	matrix4 C, P, iC;

	camera(float3 p, float3 d):
		fv(90.0f),
		pos(p)
	{
		rot = float2{asin(d.x / cos(d.y)), asin(d.y)};
		const float3 forward = norm(d);
		const float3 right = norm(float3{-forward.z, 0.0, forward.x});
		const float3 up = norm(float3{-forward.x * forward.y, (forward.z * forward.z) + (forward.x * forward.x), -forward.z * forward.y});
		C = matrix4{
			float4{right.x, up.x, forward.x, 0.0},
			float4{right.y, up.y, forward.y, 0.0},
			float4{right.z, up.z, forward.z, 0.0},
			float4{-dot(right, p), -dot(up, p), -dot(forward, p), 1.0}
		};
		P = matrix4{
			float4{asp, 0.0f, 0.0f, 0.0f},
			float4{0.0f, 1.0f, 0.0f, 0.0f},
			float4{0.0f, 0.0f, pm1, -1.0f},
			float4{0.0f, 0.0f, pm2, 0.0f}
		};
		iC = inv(C);
	};

	camera(float3 p, float3 d, float FOV):
		fv(FOV),
		pos(p)
	{
		rot = float2{asin(d.x / cos(d.y)), asin(d.y)};
		const float3 forward = norm(d);
		const float3 right = norm(float3{-forward.z, 0.0, forward.x});
		const float3 up = norm(float3{-forward.x * forward.y, (forward.z * forward.z) + (forward.x * forward.x), -forward.z * forward.y});
		C = matrix4{
			float4{right.x, up.x, forward.x, 0.0},
			float4{right.y, up.y, forward.y, 0.0},
			float4{right.z, up.z, forward.z, 0.0},
			float4{-dot(right, p), -dot(up, p), -dot(forward, p), 1.0}
		};
		const float t = 1.0f / tan((fv * (float)PI) / 360.0f);
		P = matrix4{
			float4{asp * t, 0.0f, 0.0f, 0.0f},
			float4{0.0f, t, 0.0f, 0.0f},
			float4{0.0f, 0.0f, pm1,-1.0f},
			float4{0.0f, 0.0f, pm2, 0.0f}
		};
		iC = inv(C);
	};

	inline void movefov(float FOV) {
		fv += FOV;
		const float t = 1.0f / tan((fv * (float)PI) / 360.0f);
		P.x.x = asp * t;
		P.y.y = t;
	}

	inline void setfov(float FOV) {
		fv = FOV;
		const float t = 1.0f / tan((fv * (float)PI) / 360.0f);
		P.x.x = asp * t;
		P.y.y = t;
	}

	inline void place(float3 p, float3 d) {
		pos = p;
		rot = float2{asin(d.x / cos(d.y)), asin(d.y)};
		const float3 forward = norm(d);
		const float3 right = norm(float3{-forward.z, 0.0, forward.x});
		const float3 up = norm(float3{-forward.x * forward.y, (forward.z * forward.z) + (forward.x * forward.x), -forward.z * forward.y});
		C.x.x = right.x, C.x.y = up.x, C.x.z = forward.x,
		C.y.x = right.y, C.y.y = up.y, C.y.z = forward.y,
		C.z.x = right.z, C.z.y = up.z, C.z.z = forward.z,
		C.w.x = -dot(right, p), C.w.y = -dot(up, p), C.w.z = -dot(forward, p);
		iC = inv(C);
	}
	
	inline void move(float3 v, float2 d) {
		rot += d;
		const float k = cos(rot.y);
		const float u = (k < 0) ? -1.0f : 1.0f;
		const float3 forward = norm(float3{k * sin(rot.x), sin(rot.y), k * cos(rot.x)});
		const float3 right = norm(float3{-forward.z, 0.0, forward.x}) * u;
		const float3 up = norm(float3{-forward.x * forward.y, (forward.z * forward.z) + (forward.x * forward.x), -forward.z * forward.y}) * u;
		pos += (right * v.x) + (up * v.y) + (forward * v.z);
		C.x.x = right.x, C.x.y = up.x, C.x.z = forward.x,
		C.y.x = right.y, C.y.y = up.y, C.y.z = forward.y,
		C.z.x = right.z, C.z.y = up.z, C.z.z = forward.z,
		C.w.x = -dot(right, pos), C.w.y = -dot(up, pos), C.w.z = -dot(forward, pos);
		iC = inv(C);
	}
};