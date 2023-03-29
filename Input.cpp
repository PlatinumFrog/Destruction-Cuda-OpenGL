#include "Input.h"

const Uint8 *Input::keyState;
Uint32 Input::m;
float Input::mx, Input::my, Input::lx, Input::ly, Input::dx, Input::dy;
bool Input::lb, Input::mb, Input::rb;

camera Input::cam(float3{0.0f,0.0f,100.0f}, float3{0.01f,-1.0f,0.0000001f});

void Input::update() {

	keyState = SDL_GetKeyboardState(NULL);
	lx = mx, ly = my;
	int x, y;
	m = SDL_GetMouseState(&x, &y);
	mx = (float)x, my = (float)(1080 - y);
	dx = mx - lx, dy = my - ly;
	lb = (m & 1), mb = (m & 2), rb = (m & 4);

	if(rb){
		const float spd = keyState[SDL_SCANCODE_LSHIFT] ? 0.001f : 2.0f;
		cam.move(
			float3{
				(keyState[SDL_SCANCODE_D] * spd) - (keyState[SDL_SCANCODE_A] * spd),
				(keyState[SDL_SCANCODE_Q] * spd) - (keyState[SDL_SCANCODE_E] * spd),
				(keyState[SDL_SCANCODE_S] * spd) - (keyState[SDL_SCANCODE_W] * spd)
			},
			float2{((cos(cam.rot.y) >= 0) ? 0.005f : -0.005f) * dx, -0.005f * dy}
		);
	}
	else if (mb) cam.movefov(0.05f * dy);

#ifdef _DEBUG
	std::cout << (keyState[SDL_SCANCODE_W] ? "W" : "") << (keyState[SDL_SCANCODE_A] ? "A" : "") << (keyState[SDL_SCANCODE_S] ? "S" : "") << (keyState[SDL_SCANCODE_D] ? "D" : "") << (keyState[SDL_SCANCODE_Q] ? "Q" : "") << (keyState[SDL_SCANCODE_E] ? "E" : "") << ((keyState[SDL_SCANCODE_W] || keyState[SDL_SCANCODE_A] || keyState[SDL_SCANCODE_S] || keyState[SDL_SCANCODE_D] || keyState[SDL_SCANCODE_Q] || keyState[SDL_SCANCODE_E]) ? "\n" : "");
#endif
};
	


bool Input::key(SDL_Scancode s) {
	return (bool)keyState[s];
}
