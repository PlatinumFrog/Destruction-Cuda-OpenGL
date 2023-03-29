#include "Shader.h"

std::map<GLuint, shader> Shaders::shaders;

GLuint Shaders::compileShader(const std::string& vertexFileName, const std::string& geometryFileName, const std::string& fragmentFileName) {

	std::cout << "Creating new shader program with " << vertexFileName << ", " << geometryFileName << ", and " << fragmentFileName << ":\n";

	std::cout << "\tOpening " << vertexFileName << "...\n";

	std::ifstream vertexFile("shaders/" + vertexFileName, std::ios::binary);
	std::string vertexString((std::istreambuf_iterator<char>(vertexFile)), (std::istreambuf_iterator<char>()));
	vertexFile.close();

	std::cout << "\t\tCreating Vertex Shader... \n";
	GLuint vertShaderID = glCreateShader(GL_VERTEX_SHADER);

	std::cout << "\t\tLoading Code... \n";
	const GLchar* vString = vertexString.c_str();
	glShaderSource(vertShaderID, 1, &vString, NULL);

	std::cout << "\t\tCompiling...\n\n";
	glCompileShader(vertShaderID);
	
#ifdef _DEBUG
	// DEBUG
	GLint success;
	GLchar log[512];
	
	glGetShaderiv(vertShaderID, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertShaderID, 512, NULL, log);
		std::cout << "ERROR: Vertex shader " << vertShaderID << " failed to compile!\n_________________________________________\n" << log << std::endl;
		return 0;
	}
	else std::cout << '\t' << vertexFileName << " successfully compiled with ID: " << vertShaderID << "\n\n";

#endif

	std::cout << "\tOpening " << geometryFileName << "...\n";

	std::ifstream geometryFile("shaders/" + geometryFileName, std::ios::binary);
	std::string geometryString((std::istreambuf_iterator<char>(geometryFile)), (std::istreambuf_iterator<char>()));
	geometryFile.close();
	std::cout << "\t\tCreating Geometry Shader... \n";
	GLuint geomShaderID = glCreateShader(GL_GEOMETRY_SHADER);

	std::cout << "\t\tLoading Code... \n";
	const GLchar* gString = geometryString.c_str();
	glShaderSource(geomShaderID, 1, &gString, NULL);

	std::cout << "\t\tCompiling...\n\n";
	glCompileShader(geomShaderID);

#ifdef _DEBUG
	// DEBUG

	glGetShaderiv(geomShaderID, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(geomShaderID, 512, NULL, log);
		std::cout << "ERROR: Vertex shader " << geomShaderID << " failed to compile!\n_________________________________________\n" << log << std::endl;
		return 0;
	}
	else std::cout << '\t' << geometryFileName << " successfully compiled with ID: " << geomShaderID << "\n\n";

#endif

	std::cout << "\tOpening " << fragmentFileName << "...\n";

	std::ifstream fragmentFile("shaders/" + fragmentFileName, std::ios::binary);
	std::string fragmentString((std::istreambuf_iterator<char>(fragmentFile)), (std::istreambuf_iterator<char>()));
	fragmentFile.close();
	std::cout << "\t\tCreating Fragment Shader...\n";
	GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	std::cout << "\t\tLoading Code...\n";
	const GLchar* fString = fragmentString.c_str();
	glShaderSource(fragShaderID, 1, &fString, NULL);

	std::cout << "\t\tCompiling...\n\n";
	glCompileShader(fragShaderID);

#ifdef _DEBUG
	// DEBUG

	glGetShaderiv(fragShaderID, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragShaderID, 512, NULL, log);
		std::cout << "ERROR: Fragment shader " << fragShaderID << " failed to compile!\n___________________________________________\n" << log << std::endl;
		return 0;
	}
	else {
		std::cout << '\t' << fragmentFileName << " successfully compiled with ID: " << fragShaderID << ".\n\n";
	}

#endif

	std::cout << "\tLinking shader program...\n";
	GLuint shaderID = glCreateProgram();

	std::cout << "\t\tAttaching " << vertexFileName << " ID: " << vertShaderID << "...\n";
	glAttachShader(shaderID, vertShaderID);

	std::cout << "\t\tAttaching " << geometryFileName << " ID: " << geomShaderID << "...\n";
	glAttachShader(shaderID, geomShaderID);

	std::cout << "\t\tAttaching " << fragmentFileName << " ID: " << fragShaderID << "...\n";
	glAttachShader(shaderID, fragShaderID);

	std::cout << "\t\tLinking " << vertexFileName << ", " << geometryFileName << ", and " << fragmentFileName << "...\n\n";
	glLinkProgram(shaderID);

#ifdef _DEBUG
	// DEBUG
	glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderID, 512, NULL, log);
		std::cout << "ERROR: Shader Program " << shaderID << " has failed to link!\n" << log << std::endl;
		return 0;
	}
	else {
		std::cout << '\t' << vertexFileName << ", " << geometryFileName << ", and " << fragmentFileName << " were successfully linked.\n\nShader Program Succesfully created with ID: " << shaderID << "\n\n";
	}
#endif

	glDeleteShader(vertShaderID);
	glDeleteShader(geomShaderID);
	glDeleteShader(fragShaderID);

	shaders[shaderID] = {vertexFileName, vertexString, geometryFileName, geometryString, fragmentFileName, fragmentString};

	return shaderID;
}

GLuint Shaders::compileShader(const GLuint ID){

	if(shaders.count(ID)){

		shader& s = shaders[ID];

		std::string vertexString = "";
		while (vertexString == "") {
			std::ifstream vertexFile("shaders/" + s.vertFile, std::ios::binary);
			vertexString = std::string((std::istreambuf_iterator<char>(vertexFile)), (std::istreambuf_iterator<char>()));
			vertexFile.close();
		}
		std::string geometryString = "";
		while (geometryString == "") {
			std::ifstream geometryFile("shaders/" + s.geomFile, std::ios::binary);
			geometryString = std::string((std::istreambuf_iterator<char>(geometryFile)), (std::istreambuf_iterator<char>()));
			geometryFile.close();
		}
		std::string fragmentString = "";
		while (fragmentString == "") {
			std::ifstream fragmentFile("shaders/" + s.fragFile, std::ios::binary);
			fragmentString = std::string((std::istreambuf_iterator<char>(fragmentFile)), (std::istreambuf_iterator<char>()));
			fragmentFile.close();
		}
		bool vertChange = s.vertString != vertexString,
			geomChange = s.geomString != geometryString,
			fragChange = s.fragString != fragmentString;

		if(vertChange || geomChange || fragChange) {

			if (vertChange) s.vertString = vertexString;
			if (geomChange) s.geomString = geometryString;
			if (fragChange) s.fragString = fragmentString;

			GLint success = 0;
			GLchar log[512];

			std::cout << "Changes have been made to: ";
			if (vertChange) std::cout << "[" << s.vertFile << "]";
			if (geomChange) std::cout << "[" << s.geomFile << "]";
			if (fragChange) std::cout << "[" << s.fragFile << "]";
			
			glDeleteProgram(ID);
			
			std::cout << "\n\tRecompiling " << s.vertFile << "...\n";
			GLuint vertShaderID = glCreateShader(GL_VERTEX_SHADER);
			const GLchar* vString = vertexString.c_str();
			glShaderSource(vertShaderID, 1, &vString, NULL);
			glCompileShader(vertShaderID);
			glGetShaderiv(vertShaderID, GL_COMPILE_STATUS, &success);
			if(success) std::cout << '\t' << s.vertFile << " successfully recompiled with ID: " << vertShaderID << "\n\n";
			else {
				glGetShaderInfoLog(vertShaderID, 512, NULL, log);
				std::cout << "ERROR: Vertex shader " << vertShaderID << " failed to compile!\n_________________________________________\n" << log << std::endl;
				return 0;
			}
			success = 0;

			std::cout << "\tRecompiling " << s.geomFile << "...\n";
			GLuint geomShaderID = glCreateShader(GL_GEOMETRY_SHADER);
			const GLchar* gString = geometryString.c_str();
			glShaderSource(geomShaderID, 1, &gString, NULL);
			glCompileShader(geomShaderID);
			glGetShaderiv(geomShaderID, GL_COMPILE_STATUS, &success);
			if(success) std::cout << '\t' << s.geomFile << " successfully compiled with ID: " << geomShaderID << ".\n\n";
			else {
				glGetShaderInfoLog(geomShaderID, 512, NULL, log);
				std::cout << "ERROR: Fragment shader " << geomShaderID << " failed to compile!\n___________________________________________\n" << log << std::endl;
				return 0;
			}
			success = 0;

			std::cout << "\tRecompiling " << s.fragFile << "...\n";
			GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
			const GLchar* fString = fragmentString.c_str();
			glShaderSource(fragShaderID, 1, &fString, NULL);
			glCompileShader(fragShaderID);

			glGetShaderiv(fragShaderID, GL_COMPILE_STATUS, &success);
			if (success) std::cout << '\t' << s.fragFile << " successfully compiled with ID: " << fragShaderID << ".\n\n";
			else {
				glGetShaderInfoLog(fragShaderID, 512, NULL, log);
				std::cout << "ERROR: Fragment shader " << fragShaderID << " failed to compile!\n___________________________________________\n" << log << std::endl;
				return 0;
			}
			success = 0;

			std::cout << "\tRelinking shader program " << ID << "...\n";
			GLuint shaderID = glCreateProgram();
			glAttachShader(shaderID, vertShaderID);
			glAttachShader(shaderID, geomShaderID);
			glAttachShader(shaderID, fragShaderID);
			glLinkProgram(shaderID);
			glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
			if(success) std::cout << '\t' << s.vertFile << ", " << s.geomFile << ", and " << s.fragFile << " were successfully relinked with program ID: " << shaderID << "\n\n";
			else {
				glGetProgramInfoLog(shaderID, 512, NULL, log);
				std::cout << "ERROR: Shader Program " << shaderID << " has failed to link!\n___________________________________________\n" << log << std::endl;
				return 0;
			}

			glDeleteShader(vertShaderID);
			glDeleteShader(geomShaderID);
			glDeleteShader(fragShaderID);
			
			shaders[shaderID] = {s.vertFile, vertexString, s.geomFile, geometryString, s.fragFile, fragmentString};
			if(shaderID != ID) shaders.erase(ID);
			return shaderID;
		} else return ID;
	}
	return 0;
}