#include "VAOnD.cuh"

template<size_t c>
VAOSphere<c>::VAOSphere():
	vaoID(0),
	shaderID(Shaders::compileShader("spheres.vert", "spheres.geom", "spheres.frag")),
	//shaderOBBID(Shaders::compileShader("boundingbox.vert", "boundingbox.geom", "boundingbox.frag")),
	enabled(false),
	position(),
	color(),
	shaderCompileToggle(true),
	shaderAnim(0.0f)
{

	// Bind the main buffers to the VAO
	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);
	position.allocate(c, 0, false);
	color.allocate(c, 1, true);
	glBindVertexArray(0);
#ifdef USE_TEXTURE
	// Load image of the celestial body
	SDL_Surface* img = IMG_Load("18.png");
	glGenTextures(1, &imgID);
	glBindTexture(GL_TEXTURE_2D, imgID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img->w, img->h, 0, GL_RGB, GL_UNSIGNED_BYTE, img->pixels);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	SDL_FreeSurface(img);
#endif
}

template<size_t c>
VAOSphere<c>::~VAOSphere() {

	glDeleteVertexArrays(1, &vaoID);
	glDeleteProgram(shaderID);
}

template<size_t c>
void VAOSphere<c>::enable() {
	enabled = true;
}

template<size_t c>
void VAOSphere<c>::disable() {
	enabled = false;
}

template<size_t c>
void VAOSphere<c>::draw(size_t s) {
#ifdef _DEBUG
	shaderID = Shaders::compileShader(shaderID);
#endif
	if(!(position.mode || color.mode)) {
		if(!enabled) enable();

		
		////glDisable(GL_DEPTH_CLAMP);
		////glDepthRange(0.0000001, 10000.0);

		////Clear and Set Viewport
		glViewport(0, 0, 1920, 1080);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//// Draw Bounding Boxes
		//glUseProgram(shaderOBBID);
		//glBindVertexArray(vaoID);

		//const matrix4 view = Input::cam.P * Input::cam.C;
		//glUniformMatrix4fv(2, 1, GL_FALSE, (const GLfloat*)&view.x.x);

		//glDrawArrays(GL_POINTS, 0, (GLsizei)((s <= c) ? s : c));

		// Draw Spheres
		glUseProgram(shaderID);
		glBindVertexArray(vaoID);

		glUniformMatrix4fv(2, 1, GL_FALSE, (const GLfloat*)&Input::cam.P.x.x);
		glUniformMatrix4fv(6, 1, GL_FALSE, (const GLfloat*)&Input::cam.C.x.x);
		//glUniformMatrix4fv(10, 1, GL_FALSE, (const GLfloat*)&Input::cam.iC.x.x);
		//glUniform1fv(14, 1, &shaderAnim);
#ifdef USE_TEXTURE

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, imgID);
#endif
		glDrawArrays(GL_POINTS, 0, (GLsizei)((s <= c) ? s : c));

	}
	shaderAnim += 0.00001f;
	if(shaderAnim >= 1.0f) shaderAnim = 0.0f;
}

template class VAOSphere<mainPointSize>;