#ifndef OPTIX_APP_H
#define OPTIX_APP_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

struct ShaderBindingTable {
	CUdeviceptr raygenRecord;
	CUdeviceptr missRecordBase;
	unsigned int missRecordStrideInBytes;
	unsigned int missRecordCount;
	CUdeviceptr hitgroupRecordBase;
	unsigned int hitgroupRecordStrideInBytes;
	unsigned int hitgroupRecordCount;
};

class OptixApp {
public:
    OptixApp();
    ~OptixApp();

    // 初始化 CUDA、OptiX 上下文
    bool initialize();

    // 通过 tinyobjloader 加载 OBJ 模型
    bool loadObj(const std::string &filename);

    bool buildAccel();

    // 设置光线追踪管线，包括创建模块、程序组、管线、和 SBT
    bool setupPipeline();

    // 启动光线追踪
    bool launch();

    // 将输出结果保存为 PPM 图像
    bool saveOutput(const std::string &filename);

    // 清理所有资源
    void cleanup();

private:
    // CUDA 与 OptiX 上下文
    CUcontext cuCtx;
    OptixDeviceContext optixContext;

    // Pipeline 和 Program Group
    OptixPipeline pipeline;
    OptixModule  raygenModule;
    OptixModule  missModule;
    OptixModule  hitModule;
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPG;
    OptixProgramGroup hitPG;
    OptixShaderBindingTable sbt;

    // 几何数据（仅存储顶点和索引，实际中还需要法线、纹理坐标等）
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // 输出图像参数及设备缓冲区
    CUdeviceptr d_outputBuffer;
    unsigned int imageWidth;
    unsigned int imageHeight;

	// 用于 GAS 构建及管理
	OptixTraversableHandle gasHandle;
	CUdeviceptr d_vertices;
	CUdeviceptr d_indices;
	CUdeviceptr d_gasOutputBuffer;

    //custom information
    CUdeviceptr d_triangleColors = 0;

	// 用 NVRTC 编译设备代码
	std::string compilePTX(const char* cuda_code);
};

#endif // OPTIX_APP_H
