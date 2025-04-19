#ifndef OPTIX_APP_H
#define OPTIX_APP_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>


class OptixApp
{
public:
	bool  initialize();
	bool  loadObj(const std::string& file);
	bool  buildAccel();
	bool  setupPipeline();
	bool  launch();
	void  cleanup();

private:
	// host data
	std::vector<float>            vertices;      // xyz …
	std::vector<unsigned int>     indices;       // i0 i1 i2 …
	std::vector<float3>           centroids;     // per‑triangle centre

	// CUDA / OptiX handles
	CUcontext                     cuCtx = 0;
	OptixDeviceContext            optixContext = nullptr;
	OptixPipeline                 pipeline = nullptr;
	OptixModule                   module = nullptr;
	OptixProgramGroup             pgRaygen = nullptr;
	OptixProgramGroup             pgMiss = nullptr;
	OptixProgramGroup             pgAnyHit = nullptr;
	OptixShaderBindingTable       sbt{};

	// geometry
	CUdeviceptr                   d_vertices = 0;
	CUdeviceptr                   d_indices = 0;
	CUdeviceptr                   d_gasOutputBuffer = 0;
	CUdeviceptr                   d_triCentre = 0;
	CUdeviceptr                   d_visBits = 0;
	OptixTraversableHandle        gasHandle = 0;
	OptixTraversableHandle compactedAccelHandle = 0;

	cudaTextureObject_t texCentroids = 0;
	// constants
	unsigned int kSMMajor = 7, kSMMinor = 0;   // change to your GPU
};

#endif // OPTIX_APP_H
