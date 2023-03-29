#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <glad.h>

struct shader {
	std::string vertFile, vertString;
	std::string geomFile, geomString;
	std::string fragFile, fragString;
};

class Shaders {
	static std::map<GLuint, shader> shaders;
public:
	static GLuint compileShader(const std::string& vertexFileName, const std::string& geometryFileName, const std::string& fragmentFileName);
	static GLuint compileShader(const GLuint ID);
};
