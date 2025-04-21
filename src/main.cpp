#include "optix_app.h"
#include <iostream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

int main(int argc, char** argv) {

	OptixApp app;
	if (!app.initialize()) {
		std::cerr << "OptiX initialization failed\n";
		return EXIT_FAILURE;
	}

	std::string objPath = "D:\\Documents\\CODES\\OptixTest\\assets\\DoublePassNIST_Mesh.obj";
	if (!app.loadObj(objPath)) {
		std::cerr << "Loading OBJ failed" << objPath << "\n";
		return EXIT_FAILURE;
	}

	if (!app.buildAccel()) {
		std::cerr << "Error buildAccel: " << objPath << "\n";
		return EXIT_FAILURE;
	}

	if (!app.setupPipeline()) {
		std::cerr << "Setting OptiX pipeline failed\n";
		return EXIT_FAILURE;
	}

	if (!app.launch()) {
		std::cerr << "Launching failed\n";
		return EXIT_FAILURE;
	}

	/*if (!app.saveOutput("output.ppm")) {
		std::cerr << "out put filed\n";
		return EXIT_FAILURE;
	}*/

	app.cleanup();
	std::cout << "file saved at output.ppm\n";
	return EXIT_SUCCESS;
}
