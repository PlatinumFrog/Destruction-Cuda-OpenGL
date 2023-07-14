#pragma once
#include <SDL.h>
#include "camera.cuh"

class Input {

	static const Uint8 *keyState;
	static Uint32 m;

public:

	static float mx, my, lx, ly, dx, dy;
	static bool lb, mb, rb;

	static camera cam;

	static void update();

	static bool key(SDL_Scancode s);

};