//------------------------------------------------------------------------------
//  OptiX 9.0 – Triangle‑to‑Triangle Visibility Matrix Demo
//  Complete, single‑file example (C++17).
//  ‑ 2025‑04‑17 – ChatGPT
//------------------------------------------------------------------------------

#include "optix_app.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <nvrtc.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>

//------------------------------------------------------------------------------
//  Helpers
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                           \
    do {                                                                             \
        cudaError_t _e = call;                                                       \
        if( _e != cudaSuccess ) {                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString( _e )                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";         \
            std::exit( 1 );                                                          \
        }                                                                            \
    } while( 0 )

#define OPTIX_CHECK( call )                                                          \
    do {                                                                             \
        OptixResult _r = call;                                                       \
        if( _r != OPTIX_SUCCESS ) {                                                  \
            std::cerr << "OptiX Error: " << _r                                      \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";         \
            std::exit( 1 );                                                          \
        }                                                                            \
    } while( 0 )

#define NVRTC_CHECK( call ) \
	do { nvrtcResult _e = call;                                        \
		 if( _e != NVRTC_SUCCESS ) {                                   \
			 std::cerr << "NVRTC Error: "                              \
					   << nvrtcGetErrorString( _e )                    \
					   << " (" << __FILE__ << ":" << __LINE__ << ")\n";\
			 std::exit( 1 ); }                                         \
	} while(0)


//------------------------------------------------------------------------------
//  Log callback
//------------------------------------------------------------------------------
static void context_log_callback(unsigned int level, const char* tag,
	const char* message, void*)
{
	std::cerr << "[OptiX][" << level << "](" << tag << ") " << message << "\n";
}

//------------------------------------------------------------------------------
//  Device code – embedded CUDA
//------------------------------------------------------------------------------
static const char* deviceCode = R"(
#include <optix.h>
#include <optix_device.h>
#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <optix.h>

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

struct LaunchParams
{
    uint32_t*               visBits;    // upper‑triangular bit‑matrix
    cudaTextureObject_t     triCentreTex;   // <‑‑ changed  // N centres
    uint32_t                triCount;   // N
    OptixTraversableHandle  tlas;
};

__constant__ LaunchParams lp;

// 1‑bit payload in slot 0
struct PayloadVis { unsigned int flag; };

extern "C" __global__ void __raygen__rg() 
{
	if (optixGetLaunchIndex().x == 0 && optixGetLaunchIndex().y == 0) {
        printf("DBG lp.triCount=%u, lp.triCentreTex=%llu, lp.visBits=%p\n",
               lp.triCount,
               (unsigned long long)lp.triCentreTex,
               (void*)lp.visBits);
    }
	
    uint32_t src   = optixGetLaunchIndex().y;   // row
    uint32_t wID   = optixGetLaunchIndex().x;   // word along the row
    uint32_t dst0  = wID << 5;                  // first dest in this word

    // position in packed buffer
    // 先算出这一行在上三角矩阵里的首位 bit 索引
	uint64_t rowStart = (uint64_t)src * lp.triCount
					  - (uint64_t)src * (src - 1) / 2;
	// 再算出这一组 32 位 word 的下标
	uint64_t wordIdx = (rowStart >> 5) + wID;

    float3  o = make_float3(tex1Dfetch<float4>(lp.triCentreTex, src));
    unsigned int bits = 0u;

    #pragma unroll
    for (int lane = 0; lane < 32; ++lane)
    {
        uint32_t dst = dst0 + lane;
        if (dst >= lp.triCount) break;
        if (dst == src) { bits |= 1u << lane; continue; }

		float3 t = make_float3(tex1Dfetch<float4>(lp.triCentreTex, dst));
        float3 d    = t - o;
        float  dist = length(d);
        float  tmin = 1e-4f;                  // 跳过自身相交
        float  tmax = dist - 1e-4f;           // 跳过目标自身
		
		//if(src%1000==0)
		//printf("centroid[%d] = (%f, %f, %f)\n", src, o.x, o.y, o.z);			

        unsigned int vis = 0u;
        optixTrace( lp.tlas, o, normalize(d),
                    tmin, tmax, 0.f,
                    0xFF,
                    0,
                    0, 1, 0, vis );

        bits |= vis << lane;
    }

    // warp‑wide OR, one store per 32 threads
    lp.visBits[wordIdx] = bits;
}

extern "C" __global__ void __miss__ms()  
{
	//printf("miss");
    optixSetPayload_0( 1u ); // visible – no geometry hit
}

extern "C" __global__ void __anyhit__ah() 
{
	printf("blocked: prim=%u\n", optixGetPrimitiveIndex());
    optixSetPayload_0( 0u ); // blocked
    optixTerminateRay();
}
)";

//------------------------------------------------------------------------------
//  initialize
//------------------------------------------------------------------------------
bool OptixApp::initialize()
{
	CUDA_CHECK(cudaFree(nullptr));   // init context

	OPTIX_CHECK(optixInit());

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	kSMMajor = prop.major; kSMMinor = prop.minor;

	OptixDeviceContextOptions options{};
	options.logCallbackFunction = context_log_callback;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));
	return true; 
}

//------------------------------------------------------------------------------
//  load OBJ – also builds centroid list
//------------------------------------------------------------------------------
bool OptixApp::loadObj(const std::string& filename)
{
	tinyobj::attrib_t                attrib;
	std::vector<tinyobj::shape_t>    shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str()))
	{
		std::cerr << warn << err << "\n";
		return false;
	}

	vertices.clear(); indices.clear();

	for (const auto& s : shapes)
	{
		size_t offset = 0;
		for (size_t f = 0; f < s.mesh.num_face_vertices.size(); ++f)
		{
			if (s.mesh.num_face_vertices[f] != 3)
			{
				std::cerr << "Non‑triangle face detected – abort\n";
				return false;
			}
			float3 vs[3];
			for (int v = 0; v < 3; ++v)
			{
				const tinyobj::index_t idx = s.mesh.indices[offset + v];
				vs[v].x = attrib.vertices[3 * idx.vertex_index + 0];
				vs[v].y = attrib.vertices[3 * idx.vertex_index + 1];
				vs[v].z = attrib.vertices[3 * idx.vertex_index + 2];

				vertices.insert(vertices.end(), { vs[v].x, vs[v].y, vs[v].z });
				indices.push_back(static_cast<unsigned int>(indices.size()));
			}
			offset += 3;
			// centroid
			centroids.push_back(make_float3((vs[0].x + vs[1].x + vs[2].x) / 3.f,
				(vs[0].y + vs[1].y + vs[2].y) / 3.f,
				(vs[0].z + vs[1].z + vs[2].z) / 3.f));
		}

		// Convert to float4 (CUDA needs 32‑bit aligned channel) ▼
		std::vector<float4> centroids4;
		centroids4.reserve(centroids.size());
		for (const auto& c : centroids)
			centroids4.push_back(make_float4(c.x, c.y, c.z, 0.f));

		// --- allocate CUDA array + upload ------------------------------------------
		cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float4>();
		cudaArray_t d_centroidArray;
		CUDA_CHECK(cudaMallocArray(&d_centroidArray, &chDesc,
			static_cast<size_t>(centroids4.size()), 1,
			cudaArrayDefault));

		CUDA_CHECK(cudaMemcpy2DToArray(d_centroidArray, 0, 0,
			centroids4.data(), sizeof(float4) * centroids4.size(),
			sizeof(float4) * centroids4.size(), 1,
			cudaMemcpyHostToDevice));

		// --- create texture object --------------------------------------------------
		cudaResourceDesc resDesc{};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_centroidArray;

		cudaTextureDesc texDesc{};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;                 // use integer index

		CUDA_CHECK(cudaCreateTextureObject(&texCentroids, &resDesc, &texDesc, nullptr));

	}
	std::cout << "Loaded " << centroids.size() << " triangles\n";
	return true;
}

//------------------------------------------------------------------------------
//  buildAccel – simple GAS
//------------------------------------------------------------------------------
bool OptixApp::buildAccel()
{
	if (vertices.empty()) { std::cerr << "No geometry\n"; return false; }

	const size_t vSize = vertices.size() * sizeof(float);
	const size_t iSize = indices.size() * sizeof(unsigned int);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vSize));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), iSize));

	CUDA_CHECK(cudaMemcpy((void*)d_vertices, vertices.data(), vSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((void*)d_indices, indices.data(), iSize, cudaMemcpyHostToDevice));

	OptixBuildInput triInput{};
	triInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triInput.triangleArray.vertexBuffers = &d_vertices;
	triInput.triangleArray.numVertices = static_cast<unsigned int>(vertices.size() / 3);
	triInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
	triInput.triangleArray.indexBuffer = d_indices;
	triInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
	triInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
	static unsigned int triFlags = OPTIX_GEOMETRY_FLAG_NONE;
	triInput.triangleArray.flags = &triFlags;
	triInput.triangleArray.numSbtRecords = 1;

	OptixAccelBuildOptions accelOpts{};
	accelOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | 
		OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
		OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
	accelOpts.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes sizes{};
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
		&accelOpts, &triInput, 1, &sizes));

	CUdeviceptr d_temp;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), sizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutputBuffer), sizes.outputSizeInBytes));

	CUdeviceptr d_compactedSize = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(size_t)));

	OptixAccelEmitDesc prop{};
	prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	prop.result = d_compactedSize;

	OPTIX_CHECK(optixAccelBuild(optixContext, 0,
		&accelOpts, &triInput, 1,
		d_temp, sizes.tempSizeInBytes,
		d_gasOutputBuffer, sizes.outputSizeInBytes,
		&gasHandle, &prop, 1));
	CUDA_CHECK(cudaFree((void*)d_temp));


	//size_t compactedSize = 0;
	//CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(d_compactedSize),
	//	sizeof(size_t), cudaMemcpyDeviceToHost));
	//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_compactedSize)));

	//CUdeviceptr d_compactedOutputBuffer  = 0;
	//CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactedOutputBuffer), compactedSize));

	//if (compactedSize < sizes.outputSizeInBytes) {
	//	optixAccelCompact(
	//		optixContext, 0,
	//		gasHandle,
	//		d_compactedOutputBuffer, compactedSize,
	//		&compactedAccelHandle);
	//}

	//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutputBuffer)));
	//gasHandle = compactedAccelHandle;   // 用压缩后的 handle 替换
	//d_gasOutputBuffer = d_compactedOutputBuffer;

	// upload centroids
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triCentre), centroids.size() * sizeof(float3)));
	CUDA_CHECK(cudaMemcpy((void*)d_triCentre, centroids.data(),
		centroids.size() * sizeof(float3), cudaMemcpyHostToDevice));

	std::cout << "GAS built (" << sizes.outputSizeInBytes / 1e6 << " MB)\n";
	//std::cout << "Compacted:" << compactedSize / 1e6 << " MB)\n";
	return true;
}

//------------------------------------------------------------------------------
//  compile PTX via NVRTC
//------------------------------------------------------------------------------
static std::string compilePTX(const char* code, int sm)
{
	nvrtcProgram prog;
	NVRTC_CHECK(nvrtcCreateProgram(&prog, code, "vis.cu", 0, nullptr, nullptr));

	const std::string gpuArch = "--gpu-architecture=compute_" + std::to_string(sm);
	std::vector<std::string> include_paths = {
		"--use_fast_math",
		"-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include",
		"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include",
		"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/SDK",
		"--generate-line-info",
		"-default-device"
	};


	std::vector<const char*> opts;
	opts.push_back(gpuArch.c_str());
	for (const auto& s : include_paths) opts.push_back(s.c_str());

	nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

	size_t logSize; nvrtcGetProgramLogSize(prog, &logSize);
	if (logSize) {
		std::string log(logSize, '\0'); nvrtcGetProgramLog(prog, &log[0]);
		std::cout << log << "\n";
	}

	if (res != NVRTC_SUCCESS) { std::cerr << "NVRTC compile failed\n"; std::exit(1); }

	size_t ptxSize; nvrtcGetPTXSize(prog, &ptxSize);
	std::string ptx(ptxSize, '\0'); nvrtcGetPTX(prog, &ptx[0]);
	nvrtcDestroyProgram(&prog);
	return ptx;
}

//------------------------------------------------------------------------------
//  setupPipeline
//------------------------------------------------------------------------------
bool OptixApp::setupPipeline()
{
	// 1. compile PTX
	const std::string ptx = compilePTX(deviceCode, kSMMajor * 10 + kSMMinor);

	// 2. Payload semantics – 1 slot, caller R/W + AH/MS write
	unsigned int sem[1] = { OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE |
							 OPTIX_PAYLOAD_SEMANTICS_AH_WRITE |
							 OPTIX_PAYLOAD_SEMANTICS_MS_WRITE };
	OptixPayloadType pType{ 1, sem };

	// 3. Module options
	OptixModuleCompileOptions mco{};
	mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	mco.numPayloadTypes = 1;
	mco.payloadTypes = &pType;

	// 4. Pipeline options
	OptixPipelineCompileOptions pco{};
	pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pco.numAttributeValues = 2;
	pco.pipelineLaunchParamsVariableName = "lp"; // inside device code we used lp via constant
	pco.numPayloadValues = 0;     // because we use payloadType API
	pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	pco.exceptionFlags |= OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

	char log[2048]; size_t logSize;
	logSize = sizeof(log);
	OPTIX_CHECK(optixModuleCreate(optixContext, &mco, &pco,
		ptx.c_str(), ptx.size(),
		log, &logSize, &module));

	// 5. Program groups
	OptixProgramGroupOptions pgo{};

	OptixProgramGroupDesc descRG{}; descRG.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	descRG.raygen.module = module;
	descRG.raygen.entryFunctionName = "__raygen__rg";

	OptixProgramGroupDesc descMS{}; descMS.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	descMS.miss.module = module;
	descMS.miss.entryFunctionName = "__miss__ms";

	OptixProgramGroupDesc descAH{}; descAH.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	descAH.hitgroup.moduleAH = module;
	descAH.hitgroup.entryFunctionNameAH = "__anyhit__ah";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &descRG, 1, &pgo, log, &logSize, &pgRaygen));
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &descMS, 1, &pgo, log, &logSize, &pgMiss));
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &descAH, 1, &pgo, log, &logSize, &pgAnyHit));

	// 6. Pipeline
	OptixProgramGroup pgs[] = { pgRaygen, pgMiss, pgAnyHit };
	OptixPipelineLinkOptions linkOpts{}; linkOpts.maxTraceDepth = 1;

	OPTIX_CHECK(optixPipelineCreate(optixContext, &pco, &linkOpts,
		pgs, 3, log, &logSize, &pipeline));

	//OPTIX_CHECK(optixPipelineSetStackSize(
	//	pipeline,
	//	        /* directCallableFromTraversalStackSize */ 0,
	//	        /* directCallableFromStateStackSize     */ 0,
	//	        /* continuationStackSize                */ 2 * 1024,
	//	        /* maxTraversableDepth                  */ 1
	//	 ));

	// 7. SBT
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

	SbtRecord rgRec{}, msRec{}, ahRec{};
	OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, &rgRec));
	OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss, &msRec));
	OPTIX_CHECK(optixSbtRecordPackHeader(pgAnyHit, &ahRec));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.raygenRecord), sizeof(SbtRecord)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.missRecordBase), sizeof(SbtRecord)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroupRecordBase), sizeof(SbtRecord)));

	CUDA_CHECK(cudaMemcpy((void*)sbt.raygenRecord, &rgRec, sizeof(SbtRecord), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((void*)sbt.missRecordBase, &msRec, sizeof(SbtRecord), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((void*)sbt.hitgroupRecordBase, &ahRec, sizeof(SbtRecord), cudaMemcpyHostToDevice));

	sbt.missRecordStrideInBytes = sizeof(SbtRecord); sbt.missRecordCount = 1;
	sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord); sbt.hitgroupRecordCount = 1;

	return true;
}

//------------------------------------------------------------------------------
//  launch – allocates visBits buffer, fills params, runs optixLaunch
//------------------------------------------------------------------------------
bool OptixApp::launch()
{
	const uint32_t N = static_cast<uint32_t>(centroids.size());

	// Vis matrix words (upper‑triangle)
	const uint64_t totalWords = ((uint64_t)N * (N + 31ull)) >> 6;  // N*(N+31)/64
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_visBits), totalWords * sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset((void*)d_visBits, 0, totalWords * sizeof(uint32_t)));

	// Launch params
	struct Params { 
		uint32_t* visBits; 
		cudaTextureObject_t    triCentreTex;   // <‑‑ changed
		uint32_t triCount; 
		OptixTraversableHandle tlas; } 
	params;
	params.visBits = (uint32_t*)d_visBits;
	params.triCentreTex = texCentroids;
	params.triCount = N;
	params.tlas = gasHandle;

	CUdeviceptr d_params; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(params)));
	CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(params), cudaMemcpyHostToDevice));

	uint32_t wordsPerRow = (N + 31) >> 5;

	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);


	OPTIX_CHECK(optixLaunch(pipeline, 0, d_params, sizeof(params), &sbt,
		wordsPerRow, N, 1));

	cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms, start, stop);
	std::cout << "optixLaunch: " << ms << " ms\n";

	cudaEventDestroy(start); cudaEventDestroy(stop);

	// read back & save here
	std::vector<uint32_t> hostBits(totalWords);
	CUDA_CHECK(cudaMemcpy(hostBits.data(), (void*)d_visBits,
		totalWords * sizeof(uint32_t),
		cudaMemcpyDeviceToHost));

	std::ofstream fout("vis.bin", std::ios::binary);
	fout.write(reinterpret_cast<char*>(hostBits.data()),
		hostBits.size() * sizeof(uint32_t));
	fout.close();



	CUDA_CHECK(cudaFree((void*)d_params));
	return true;
}

//------------------------------------------------------------------------------
//  cleanup
//------------------------------------------------------------------------------
void OptixApp::cleanup()
{
	if (d_visBits)          CUDA_CHECK(cudaFree((void*)d_visBits));
	if (d_triCentre)        CUDA_CHECK(cudaFree((void*)d_triCentre));
	if (d_vertices)         CUDA_CHECK(cudaFree((void*)d_vertices));
	if (d_indices)          CUDA_CHECK(cudaFree((void*)d_indices));
	if (d_gasOutputBuffer)  CUDA_CHECK(cudaFree((void*)d_gasOutputBuffer));

	if (sbt.raygenRecord)   CUDA_CHECK(cudaFree((void*)sbt.raygenRecord));
	if (sbt.missRecordBase) CUDA_CHECK(cudaFree((void*)sbt.missRecordBase));
	if (sbt.hitgroupRecordBase) CUDA_CHECK(cudaFree((void*)sbt.hitgroupRecordBase));

	if (pipeline)     OPTIX_CHECK(optixPipelineDestroy(pipeline));
	if (pgRaygen)     OPTIX_CHECK(optixProgramGroupDestroy(pgRaygen));
	if (pgMiss)       OPTIX_CHECK(optixProgramGroupDestroy(pgMiss));
	if (pgAnyHit)     OPTIX_CHECK(optixProgramGroupDestroy(pgAnyHit));
	if (module)       OPTIX_CHECK(optixModuleDestroy(module));
	if (optixContext) OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
}


