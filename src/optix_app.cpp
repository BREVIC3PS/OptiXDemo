#include "optix_app.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "nvrtc.h"
//#include "optix_device.h"


// Log callback function for OptiX
static void context_log_callback(unsigned int level,
	const char* tag,
	const char* message,
	void* /*cbdata*/)
{
	std::cerr << "[OptiX][Level " << level << "][" << tag << "]: " << message << "\n";
}

OptixApp::OptixApp()
	: cuCtx(0),
	optixContext(nullptr),
	pipeline(nullptr),
	raygenModule(nullptr),
	missModule(nullptr),
	hitModule(nullptr),
	raygenPG(nullptr),
	missPG(nullptr),
	hitPG(nullptr),
	d_outputBuffer(0),
	imageWidth(6400),
	imageHeight(4800),
	gasHandle(0),
	d_vertices(0),
	d_indices(0),
	d_gasOutputBuffer(0)
{
	std::memset(&sbt, 0, sizeof(sbt));
}

OptixApp::~OptixApp()
{
	cleanup();
}

bool OptixApp::initialize()
{
	// Initialize CUDA
	cudaError_t cudaStatus = cudaFree(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << "\n";
		return false;
	}
	// Use current CUDA context by default
	cuCtx = 0;

	// Initialize OptiX
	if (optixInit() != OPTIX_SUCCESS) {
		std::cerr << "optixInit() failed\n";
		return false;
	}

	// Example dummy usage (can be removed)
	OptixClusterAccelBuildInput a;
	a.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES;
	a.triangles.maxTotalVertexCount = 1;

	// Create OptiX device context
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = context_log_callback;
	options.logCallbackLevel = 4;

	if (optixDeviceContextCreate(cuCtx, &options, &optixContext) != OPTIX_SUCCESS) {
		std::cerr << "optixDeviceContextCreate() failed\n";
		return false;
	}

	return true;
}

bool OptixApp::loadObj(const std::string& filename)
{
	// Load OBJ by tinyobjloader
	tinyobj::attrib_t                 attrib;
	std::vector<tinyobj::shape_t>     shapes;
	std::vector<tinyobj::material_t>  materials;
	std::string                       warn, err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
	if (!warn.empty()) std::cerr << "tinyobjloader warning: " << warn << "\n";
	if (!err.empty())  std::cerr << "tinyobjloader error: " << err << "\n";
	if (!ret)          return false;

	vertices.clear();
	indices.clear();

	// Only process triangle faces
	for (size_t s = 0; s < shapes.size(); ++s) {
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
			int fv = shapes[s].mesh.num_face_vertices[f];
			if (fv != 3) {
				std::cerr << "Detected non‑triangle face, only triangles are supported!\n";
				return false;
			}
			for (int v = 0; v < fv; ++v) {
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				float vx = attrib.vertices[3 * idx.vertex_index + 0];
				float vy = attrib.vertices[3 * idx.vertex_index + 1];
				float vz = attrib.vertices[3 * idx.vertex_index + 2];
				vertices.push_back(vx);
				vertices.push_back(vy);
				vertices.push_back(vz);
				indices.push_back(static_cast<unsigned int>(indices.size()));
			}
			index_offset += fv;
		}
	}
	std::cout << "OBJ vertices: " << vertices.size() / 3
		<< ", triangles: " << indices.size() / 3 << "\n";
	return true;
}

bool OptixApp::buildAccel()
{
	if (vertices.empty() || indices.empty()) {
		std::cerr << "No geometry data, cannot build AS!\n";
		return false;
	}

	// Generate a random color for every triangle
	int numTriangles = static_cast<int>(indices.size() / 3);
	std::vector<float3> triangleColorsHost(numTriangles);

	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	for (int i = 0; i < numTriangles; ++i) {
		float r = static_cast<float>(std::rand()) / RAND_MAX;
		float g = static_cast<float>(std::rand()) / RAND_MAX;
		float b = static_cast<float>(std::rand()) / RAND_MAX;
		triangleColorsHost[i] = make_float3(r, g, b);
	}

	cudaMalloc(reinterpret_cast<void**>(&d_triangleColors),
		triangleColorsHost.size() * sizeof(float3));
	cudaMemcpy(reinterpret_cast<void*>(d_triangleColors),
		triangleColorsHost.data(),
		triangleColorsHost.size() * sizeof(float3),
		cudaMemcpyHostToDevice);

	size_t verticesSizeInBytes = vertices.size() * sizeof(float);
	size_t indicesSizeInBytes = indices.size() * sizeof(unsigned int);
	cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSizeInBytes);
	cudaMemcpy(reinterpret_cast<void*>(d_vertices),
		vertices.data(),
		verticesSizeInBytes,
		cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void**>(&d_indices), indicesSizeInBytes);
	cudaMemcpy(reinterpret_cast<void*>(d_indices),
		indices.data(),
		indicesSizeInBytes,
		cudaMemcpyHostToDevice);

	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	unsigned int triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.flags = &triangleFlags;
	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.numVertices = static_cast<unsigned int>(vertices.size() / 3);
	triangleInput.triangleArray.vertexBuffers = &d_vertices;
	triangleInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
	triangleInput.triangleArray.indexBuffer = d_indices;
	triangleInput.triangleArray.indexStrideInBytes = 3 * sizeof(unsigned int);

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes bufferSizes;
	if (optixAccelComputeMemoryUsage(optixContext,
		&accelOptions,
		&triangleInput,
		1,
		&bufferSizes) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to compute memory usage for AS build\n";
		return false;
	}

	CUdeviceptr d_scratch;
	cudaMalloc(reinterpret_cast<void**>(&d_scratch), bufferSizes.tempSizeInBytes);
	cudaMalloc(reinterpret_cast<void**>(&d_gasOutputBuffer), bufferSizes.outputSizeInBytes);

	if (optixAccelBuild(optixContext,
		0,
		&accelOptions,
		&triangleInput,
		1,
		d_scratch,
		bufferSizes.tempSizeInBytes,
		d_gasOutputBuffer,
		bufferSizes.outputSizeInBytes,
		&gasHandle,
		nullptr,
		0) != OPTIX_SUCCESS)
	{
		std::cerr << "AS build failed!\n";
		cudaFree(reinterpret_cast<void*>(d_scratch));
		return false;
	}
	cudaFree(reinterpret_cast<void*>(d_scratch));
	std::cout << "AS built successfully\n";
	return true;
}

// Compile CUDA device code with NVRTC and return PTX
std::string OptixApp::compilePTX(const char* cuda_code)
{
	nvrtcProgram prog;
	nvrtcResult  res = nvrtcCreateProgram(&prog,
		cuda_code,
		"raytrace.cu",
		0,
		nullptr,
		nullptr);
	if (res != NVRTC_SUCCESS) {
		std::cerr << "nvrtcCreateProgram failed\n";
		return "";
	}

	const char* opts[] = { "--gpu-architecture=compute_70",
						   "-G",
						   "-lineinfo",
						   "--include-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include",
						   "--include-path=C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include",
						   "--include-path=C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/SDK",
						   "--use_fast_math" };
	res = nvrtcCompileProgram(prog, 7, opts);

	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);
	std::string log(logSize, '\0');
	nvrtcGetProgramLog(prog, &log[0]);

	if (res != NVRTC_SUCCESS) {
		std::cerr << "NVRTC compile error:\n" << log << "\n";
		nvrtcDestroyProgram(&prog);
		return "";
	}

	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);
	std::string ptx(ptxSize, '\0');
	nvrtcGetPTX(prog, &ptx[0]);
	nvrtcDestroyProgram(&prog);

	if (!log.empty()) std::cout << "NVRTC log:\n" << log << "\n";
	return ptx;
}

// ---------------- CUDA device code (unchanged except comments) ----------------
const char* raytracing_cuda_code = R"(
#include <optix.h>
#include <optix_device.h>
#include <cuda/helpers.h>

extern "C" __device__ unsigned int g_hitCount  = 0;
extern "C" __device__ unsigned int g_missCount = 0;

extern "C"{

struct LaunchParams
{
    float4*               outputBuffer;
    unsigned int          imageWidth;
    unsigned int          imageHeight;
    OptixTraversableHandle handle;
    unsigned int*         hitCount;
    unsigned int*         missCount;
    float3*               triangleColors;
};

__constant__ LaunchParams launchParams;
}

extern "C" {

extern "C" __global__ void __raygen__rg() __attribute__((optix_payload_type(0)))
{
    const uint3 launch_idx  = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    unsigned int ix = launch_idx.x;
    unsigned int iy = launch_idx.y;

    // Normalized screen coordinates [0,1]
    float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(launch_dims.x);
    float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(launch_dims.y);
    float aspect_ratio = static_cast<float>(launch_dims.x) / launch_dims.y;

    // Camera parameters
    float3 eye    = make_float3(5.0f, 17.0f, 0.0f);
    float3 lookat = make_float3(0.0f,  0.0f, 0.0f);
    float3 up     = make_float3(0.0f,  0.0f, 1.0f);

    // View basis
    float3 forward = normalize(lookat - eye);
    float3 right   = normalize(cross(forward, up));
    float3 true_up = cross(right, forward);

    // 90° FOV → screen plane height = 2
    float sensor_height = 2.0f;
    float sensor_width  = aspect_ratio * sensor_height;

    // Map u,v to [-1,1] pixel plane
    float px = (2.0f * u - 1.0f) * (sensor_width  * 0.5f);
    float py = (1.0f - 2.0f * v) * (sensor_height * 0.5f);

    float3 pixel_world  = eye + forward + px * right + py * true_up;
    float3 ray_origin   = eye;
    float3 ray_direction = normalize(pixel_world - eye);

    unsigned int p0 = 1;  // dummy payload

    optixTrace(launchParams.handle,
               ray_origin,
               ray_direction,
               0.001f,
               1e16f,
               0.0f,
               255,
               OPTIX_RAY_FLAG_NONE,
               0,
               1,
               0,
               p0);
}

extern "C" __global__ void __miss__ms()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    unsigned int pixel_index = launch_idx.x + launch_idx.y * launchParams.imageWidth;

    launchParams.outputBuffer[pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    atomicAdd(launchParams.missCount, 1u);
}

extern "C" __global__ void __closesthit__ch()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    unsigned int pixel_index = launch_idx.x + launch_idx.y * launchParams.imageWidth;

    unsigned int primIndex = optixGetPrimitiveIndex();

    float3 color = launchParams.triangleColors[primIndex];

    launchParams.outputBuffer[pixel_index] = make_float4(color.x, color.y, color.z, 1.0f);
    atomicAdd(launchParams.hitCount, 1u);
}

})";
// ------------------------------------------------------------------------------

bool OptixApp::setupPipeline()
{
	// Compile PTX
	std::string ptx_code = compilePTX(raytracing_cuda_code);
	if (ptx_code.empty()) {
		std::cerr << "Device code compilation failed\n";
		return false;
	}

	unsigned int semantics[1] = { OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE };
	OptixPayloadType payloadType;
	payloadType.numPayloadValues = 1;
	payloadType.payloadSemantics = semantics;

	// Module & pipeline compile options
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	moduleCompileOptions.payloadTypes = &payloadType;
	moduleCompileOptions.numPayloadTypes = 1;

	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	pipelineCompileOptions.numPayloadValues = 0;

	char   log[2048];
	size_t logSize = sizeof(log);

	// Raygen module
	if (optixModuleCreate(optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptx_code.c_str(),
		ptx_code.size(),
		log,
		&logSize,
		&raygenModule) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to create raygen module:\n" << log << "\n";
		return false;
	}
	missModule = raygenModule;
	hitModule = raygenModule;

	// Raygen program group
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc    raygenDesc = {};
	raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenDesc.raygen.module = raygenModule;
	raygenDesc.raygen.entryFunctionName = "__raygen__rg";

	logSize = sizeof(log);
	if (optixProgramGroupCreate(optixContext,
		&raygenDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&raygenPG) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to create raygen program group:\n" << log << "\n";
		return false;
	}

	// Miss program group
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missDesc.miss.module = missModule;
	missDesc.miss.entryFunctionName = "__miss__ms";

	logSize = sizeof(log);
	if (optixProgramGroupCreate(optixContext,
		&missDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&missPG) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to create miss program group:\n" << log << "\n";
		return false;
	}

	// Hit program group (closest hit only)
	OptixProgramGroupDesc hitDesc = {};
	hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitDesc.hitgroup.moduleCH = hitModule;
	hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

	logSize = sizeof(log);
	if (optixProgramGroupCreate(optixContext,
		&hitDesc,
		1,
		&pgOptions,
		log,
		&logSize,
		&hitPG) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to create hit program group:\n" << log << "\n";
		return false;
	}

	OptixProgramGroup programGroups[] = { raygenPG, missPG, hitPG };
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.maxTraceDepth = 1;

	logSize = sizeof(log);
	if (optixPipelineCreate(optixContext,
		&pipelineCompileOptions,
		&linkOptions,
		programGroups,
		sizeof(programGroups) / sizeof(programGroups[0]),
		log,
		&logSize,
		&pipeline) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to create pipeline:\n" << log << "\n";
		return false;
	}

	// ----------------------------------------------------------------
	//                       Shader Binding Table
	// ----------------------------------------------------------------
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Record
	{
		char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	Record raygenRecord;
	if (optixSbtRecordPackHeader(raygenPG, &raygenRecord) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to pack raygen SBT header\n";
		return false;
	}

	Record missRecord;
	if (optixSbtRecordPackHeader(missPG, &missRecord) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to pack miss SBT header\n";
		return false;
	}

	Record hitRecord;
	if (optixSbtRecordPackHeader(hitPG, &hitRecord) != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to pack hit SBT header\n";
		return false;
	}

	CUdeviceptr d_raygenRecord;
	cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(Record));
	cudaMemcpy(reinterpret_cast<void*>(d_raygenRecord), &raygenRecord,
		sizeof(Record), cudaMemcpyHostToDevice);

	CUdeviceptr d_missRecord;
	cudaMalloc(reinterpret_cast<void**>(&d_missRecord), sizeof(Record));
	cudaMemcpy(reinterpret_cast<void*>(d_missRecord), &missRecord,
		sizeof(Record), cudaMemcpyHostToDevice);

	CUdeviceptr d_hitRecord;
	cudaMalloc(reinterpret_cast<void**>(&d_hitRecord), sizeof(Record));
	cudaMemcpy(reinterpret_cast<void*>(d_hitRecord), &hitRecord,
		sizeof(Record), cudaMemcpyHostToDevice);

	sbt.raygenRecord = d_raygenRecord;
	sbt.missRecordBase = d_missRecord;
	sbt.missRecordStrideInBytes = sizeof(Record);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = d_hitRecord;
	sbt.hitgroupRecordStrideInBytes = sizeof(Record);
	sbt.hitgroupRecordCount = 1;

	// Allocate output buffer (RGBA float4 per pixel)
	size_t imageSize = imageWidth * imageHeight * sizeof(float4);
	cudaMalloc(reinterpret_cast<void**>(&d_outputBuffer), imageSize);
	cudaMemset(reinterpret_cast<void*>(d_outputBuffer), 0, imageSize);

	return true;
}

bool OptixApp::launch()
{
	const unsigned int launchWidth = imageWidth;
	const unsigned int launchHeight = imageHeight;

	struct LaunchParams
	{
		CUdeviceptr              outputBuffer;
		unsigned int             imageWidth;
		unsigned int             imageHeight;
		OptixTraversableHandle   handle;
		unsigned int* hitCount;
		unsigned int* missCount;
		float3* triangleColors;
	} params;

	unsigned int* d_hitCounter, * d_missCounter;
	cudaMalloc(reinterpret_cast<void**>(&d_hitCounter), sizeof(unsigned int));
	cudaMalloc(reinterpret_cast<void**>(&d_missCounter), sizeof(unsigned int));
	cudaMemset(d_hitCounter, 0, sizeof(unsigned int));
	cudaMemset(d_missCounter, 0, sizeof(unsigned int));

	params.outputBuffer = d_outputBuffer;
	params.imageWidth = launchWidth;
	params.imageHeight = launchHeight;
	params.handle = gasHandle;
	params.hitCount = d_hitCounter;
	params.missCount = d_missCounter;
	params.triangleColors = reinterpret_cast<float3*>(d_triangleColors);

	CUdeviceptr d_params;
	cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams));
	cudaMemcpy(reinterpret_cast<void*>(d_params), &params,
		sizeof(LaunchParams), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	if (optixLaunch(pipeline,
		0,
		d_params,
		sizeof(LaunchParams),
		&sbt,
		launchWidth,
		launchHeight,
		1) != OPTIX_SUCCESS)
	{
		std::cerr << "optixLaunch failed\n";
		cudaFree(reinterpret_cast<void*>(d_params));
		return false;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "OptixLaunch time: " << elapsedTime << " ms\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceSynchronize();
	cudaFree(reinterpret_cast<void*>(d_params));

	unsigned int host_hitCount = 0;
	unsigned int host_missCount = 0;
	cudaMemcpy(&host_hitCount, d_hitCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_missCount, d_missCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	std::cout << "Hit rays : " << host_hitCount << "\n";
	std::cout << "Miss rays: " << host_missCount << "\n";

	cudaFree(d_hitCounter);
	cudaFree(d_missCounter);
	return true;
}

bool OptixApp::saveOutput(const std::string& filename)
{
	size_t imageSize = imageWidth * imageHeight * sizeof(float4);
	std::vector<float4> h_output(imageWidth * imageHeight);
	cudaMemcpy(h_output.data(),
		reinterpret_cast<void*>(d_outputBuffer),
		imageSize,
		cudaMemcpyDeviceToHost);

	FILE* fp = std::fopen(filename.c_str(), "wb");
	if (!fp) {
		std::cerr << "Failed to open file: " << filename << "\n";
		return false;
	}
	std::fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
	for (unsigned int i = 0; i < imageWidth * imageHeight; ++i) {
		unsigned char r = static_cast<unsigned char>(min(1.0f, max(0.0f, h_output[i].x)) * 255.99f);
		unsigned char g = static_cast<unsigned char>(min(1.0f, max(0.0f, h_output[i].y)) * 255.99f);
		unsigned char b = static_cast<unsigned char>(min(1.0f, max(0.0f, h_output[i].z)) * 255.99f);
		std::fputc(r, fp);
		std::fputc(g, fp);
		std::fputc(b, fp);
	}
	std::fclose(fp);
	std::cout << "Output saved to " << filename << "\n";
	return true;
}

void OptixApp::cleanup()
{
	// Free SBT
	if (sbt.raygenRecord) { cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));    sbt.raygenRecord = 0; }
	if (sbt.missRecordBase) { cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));  sbt.missRecordBase = 0; }
	if (sbt.hitgroupRecordBase) { cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)); sbt.hitgroupRecordBase = 0; }
	if (d_outputBuffer) { cudaFree(reinterpret_cast<void*>(d_outputBuffer));      d_outputBuffer = 0; }

	// Free geometry buffers
	if (d_vertices) { cudaFree(reinterpret_cast<void*>(d_vertices));          d_vertices = 0; }
	if (d_indices) { cudaFree(reinterpret_cast<void*>(d_indices));           d_indices = 0; }
	if (d_gasOutputBuffer) { cudaFree(reinterpret_cast<void*>(d_gasOutputBuffer));   d_gasOutputBuffer = 0; }

	// Destroy pipeline / program groups
	if (pipeline) { optixPipelineDestroy(pipeline);  pipeline = nullptr; }
	if (raygenPG) { optixProgramGroupDestroy(raygenPG); raygenPG = nullptr; }
	if (missPG) { optixProgramGroupDestroy(missPG);   missPG = nullptr; }
	if (hitPG) { optixProgramGroupDestroy(hitPG);    hitPG = nullptr; }

	if (optixContext) { optixDeviceContextDestroy(optixContext); optixContext = nullptr; }

	if (d_triangleColors) { cudaFree(reinterpret_cast<void*>(d_triangleColors)); d_triangleColors = 0; }
}
