# VulkanPG 新功能添加系统指南

## 文档说明

本文档基于 VXPG 实现经验，系统性地总结在 VulkanPG 项目中添加新路径引导功能的完整流程。旨在避免遗漏关键步骤，确保 Shader 和 C++ 两端的代码完整配套。

---

## 功能添加完整检查清单

使用本清单确保不遗漏任何步骤：

- [ ] **步骤1**: Shader端定义常量和数据结构
- [ ] **步骤2**: C++端定义对应数据结构
- [ ] **步骤3**: 创建GPU缓冲区
- [ ] **步骤4**: 绑定描述符集（Ray Tracing + Compute）
- [ ] **步骤5**: 添加描述符集布局
- [ ] **步骤6**: 实现Compute Shader（如需要）
- [ ] **步骤7**: 创建和管理Pipeline
- [ ] **步骤8**: 在路径追踪Shader中实现采样逻辑
- [ ] **步骤9**: 添加数据更新逻辑
- [ ] **步骤10**: 集成UI控制
- [ ] **步骤11**: 添加SpecializationData
- [ ] **步骤12**: 销毁资源

---

## 步骤1：Shader端定义常量和数据结构

### 文件位置
`raycommon.glsl`

### 需要添加的内容

#### 1.1 添加特化常量（Specialization Constant）

```glsl
// 在其他常量后添加（例如在 SSPG、SGM 之后）
layout (constant_id = XX) const int YOUR_FEATURE = 0;
```

**为什么要添加**：
- 特化常量允许运行时开关功能，无需重新编译Shader
- 可以在UI中动态切换不同的路径引导方法
- 编译器会根据常量值优化掉未使用的代码分支

**与什么搭配**：
- 与 `SpecializationData`（C++端）配对，通过 `VkSpecializationInfo` 传递
- 与路径追踪Shader中的条件分支配合（`if (YOUR_FEATURE == 1)`）

**起到的作用**：
- 控制功能的启用/禁用
- 实现多种方法的条件编译

**常量ID分配规则**：
- 查看已有常量的最大ID（如VXPG使用32）
- 使用下一个连续的ID号
- 记录在文档中避免冲突

---

#### 1.2 定义数据结构

```glsl
struct YourDataStructure
{
	// 根据功能需求定义字段
	// 注意：考虑内存对齐和原子操作需求
	uint field1;
	float field2;
	// ...
};
```

**为什么要添加**：
- 定义存储在GPU缓冲区中的数据格式
- 必须与C++端结构体保持内存布局一致

**设计要点**：

1. **字段类型选择**：
   - 如果需要原子操作浮点数，使用 `uint` 存储（通过 `floatBitsToUint` 转换）
   - 累积值用 `float`，计数器用 `uint`
   
2. **内存对齐**：
   - GLSL遵循std430布局规则
   - `float` 和 `uint` 是4字节对齐
   - `vec3` 会被扩展到16字节对齐（使用`vec4`或三个单独的`float`）

3. **原子操作支持**：
   - `atomicAdd`：支持 `int`、`uint`、`float`（需扩展）
   - `atomicMin/Max`：仅支持 `int`、`uint`（浮点需转换）
   - `atomicCompSwap`：支持 `int`、`uint`

**与什么搭配**：
- 与C++端的同名结构体配对
- 与Shader中的buffer声明配合（`buffer YourBuffer { YourDataStructure data[]; }`）

**起到的作用**：
- 存储算法的中间数据或统计信息
- 支持跨线程/跨帧的数据累积

**VXPG示例分析**：
```glsl
struct BoundingVoxel
{
	uint aabbMinX;  // 使用uint是因为atomicMin/Max不支持float
	// ... 其他AABB字段
	float totalIrradiance;  // 累积值可以用atomicAdd(float)
	uint sampleCount;  // 计数器用uint
};
```

---

#### 1.3 声明缓冲区绑定

```glsl
// 在 pathtracing.frag 中添加（Ray Tracing Pass）
layout(binding = XX, set = 0) buffer YourBuffer
{
	YourDataStructure data[];
} yourBuffer;

// 如果需要在 Compute Shader 中访问，也要在对应的 .comp 文件中声明
layout(binding = YY, set = 0) buffer YourBuffer
{
	YourDataStructure data[];
} yourBuffer;
```

**为什么要添加**：
- 声明Shader如何访问GPU缓冲区
- `binding` 号是Shader和C++端的桥梁

**Binding号分配规则**：
1. **Ray Tracing Pass**（pathtracing.frag）：
   - 查看现有最大binding号（如VXPG使用28）
   - 使用下一个可用号
   
2. **Compute Pass**（*.comp）：
   - Compute Shader有独立的descriptor set
   - 查看compute相关的binding（如VXPG使用7）
   - 使用下一个可用号

**与什么搭配**：
- 与C++端的 `vks::initializers::writeDescriptorSet` 调用配对
- 与C++端的 `descriptorSetLayoutBinding` 配对

**起到的作用**：
- 允许Shader读写GPU内存中的数据
- 实现跨Shader阶段的数据共享

**注意事项**：
- 不同Pass使用不同的descriptor set，binding号可以重复
- 但同一Pass内binding号必须唯一
- 建议记录在文档中避免冲突

---

## 步骤2：C++端定义对应数据结构

### 文件位置
`main.cpp`

### 需要添加的内容

```cpp
struct YourDataStructure
{
	uint32_t field1;
	float field2;
	// ... 必须与Shader端完全一致
};
```

**为什么要添加**：
- 计算缓冲区大小（`sizeof(YourDataStructure)`）
- 可能需要CPU端读取数据进行调试或统计
- 保证内存布局与Shader端一致

**内存布局一致性要求**：

1. **类型对应关系**：
   ```
   GLSL          C++
   ----          ----
   int       →   int32_t
   uint      →   uint32_t
   float     →   float
   vec2      →   float[2] 或 glm::vec2
   vec3      →   float[3] 或 glm::vec3（注意对齐）
   vec4      →   float[4] 或 glm::vec4
   ```

2. **对齐规则**：
   - GLSL使用std430布局
   - C++需要使用相同的对齐方式
   - 可以用 `#pragma pack` 或 `alignas` 控制

3. **验证方法**：
   ```cpp
   static_assert(sizeof(YourDataStructure) == EXPECTED_SIZE, "Size mismatch!");
   static_assert(offsetof(YourDataStructure, field1) == EXPECTED_OFFSET, "Offset mismatch!");
   ```

**与什么搭配**：
- 与Shader端的同名结构体配对
- 与缓冲区创建代码配合（`sizeof(YourDataStructure)`）

**起到的作用**：
- 计算GPU缓冲区的正确大小
- 作为CPU-GPU数据交换的接口

**VXPG示例**：
- Shader端和C++端的 `BoundingVoxel` 结构体完全一致
- 每个字段的类型、顺序、对齐都必须匹配

---

## 步骤3：创建GPU缓冲区

### 文件位置
`main.cpp` - `storageBuffers` 结构体和初始化函数

### 需要添加的内容

#### 3.1 在storageBuffers中声明成员

```cpp
struct {
	// ... 其他buffers
	vks::Buffer yourBuffer;  // 新增
	// ...
} storageBuffers;
```

**为什么要添加**：
- 管理GPU缓冲区的生命周期
- 集中管理所有存储缓冲区

**与什么搭配**：
- 与 `createBuffer` 调用配合
- 与 `destroy` 调用配合
- 与描述符绑定配合

---

#### 3.2 创建缓冲区

在合适的初始化函数中（通常是 `prepareStorageBuffers` 或类似函数）：

```cpp
createBuffer(
	storageBuffers.yourBuffer,                    // 目标buffer
	ELEMENT_COUNT * sizeof(YourDataStructure),    // 缓冲区大小
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,           // 用途标志
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT           // 内存属性
);
```

**参数详解**：

1. **缓冲区大小**：
   - 根据功能需求计算元素数量
   - 例如体素网格：`gridDim.x * gridDim.y * gridDim.z`
   - 例如屏幕空间：`screenWidth * screenHeight`
   - 乘以结构体大小

2. **用途标志**（VkBufferUsageFlags）：
   - `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`：作为存储缓冲区（可读写）
   - `VK_BUFFER_USAGE_TRANSFER_DST_BIT`：如果需要从CPU传输数据
   - `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`：如果需要读回CPU

3. **内存属性**（VkMemoryPropertyFlags）：
   - `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`：GPU显存（推荐，性能最好）
   - `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`：CPU可见（用于调试）
   - `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`：CPU-GPU自动同步

**为什么这样配置**：
- **DEVICE_LOCAL**：数据只在GPU端使用，放显存性能最优
- **STORAGE_BUFFER**：需要在Shader中随机读写

**与什么搭配**：
- 与Shader中的 `buffer` 声明配对
- 与描述符绑定配合

**起到的作用**：
- 在GPU显存中分配空间
- 为Shader提供数据存储

---

#### 3.3 销毁缓冲区

在清理函数中（通常是析构函数或 `cleanup` 函数）：

```cpp
storageBuffers.yourBuffer.destroy();
```

**为什么要添加**：
- 释放GPU显存
- 避免内存泄漏

**位置要求**：
- 必须在 `vkDestroyDevice` 之前调用
- 建议与创建顺序相反（LIFO）

**与什么搭配**：
- 与 `createBuffer` 对应
- 与其他资源的销毁代码并列

---

## 步骤4：绑定描述符集

### 文件位置
`main.cpp` - 描述符集更新函数

### 需要添加的内容

#### 4.1 Ray Tracing Pass 绑定

在 `setupDescriptorSets` 或类似函数中：

```cpp
vks::initializers::writeDescriptorSet(
	descriptorSets.rayTracing,                    // 目标描述符集
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,            // 描述符类型
	BINDING_NUMBER,                                // binding号（与Shader一致）
	&storageBuffers.yourBuffer.descriptor         // buffer描述符
)
```

**为什么要添加**：
- 将GPU缓冲区绑定到Shader的binding点
- 建立C++和Shader之间的数据通道

**与什么搭配**：
- 与Shader中的 `layout(binding = BINDING_NUMBER)` 对应
- 与缓冲区创建代码配合

**起到的作用**：
- 让Shader能够访问GPU缓冲区
- 在渲染时自动传递缓冲区地址

---

#### 4.2 Compute Pass 绑定（如果需要）

如果功能需要Compute Shader进行初始化或预处理：

```cpp
vks::initializers::writeDescriptorSet(
	compute.descriptorSet,                        // compute描述符集
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	COMPUTE_BINDING_NUMBER,                       // compute的binding号
	&storageBuffers.yourBuffer.descriptor
)
```

**注意事项**：
- Compute Pass有独立的descriptor set
- Binding号可以与Ray Tracing Pass重复
- 但建议使用不同的号以便调试

---

## 步骤5：添加描述符集布局

### 文件位置
`main.cpp` - Pipeline布局创建函数

### 需要添加的内容

#### 5.1 Ray Tracing布局

在 `prepareRayTracingPipeline` 或类似函数中：

```cpp
vks::initializers::descriptorSetLayoutBinding(
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,            // 描述符类型
	VK_SHADER_STAGE_FRAGMENT_BIT,                 // Shader阶段
	BINDING_NUMBER                                 // binding号
)
```

**为什么要添加**：
- 定义Pipeline可以使用哪些描述符
- 必须在Pipeline创建之前定义

**与什么搭配**：
- 与 `writeDescriptorSet` 配对
- 与Shader的binding声明对应

---

#### 5.2 Compute布局（如果需要）

在 `prepareComputePipeline` 或类似函数中：

```cpp
vks::initializers::descriptorSetLayoutBinding(
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	VK_SHADER_STAGE_COMPUTE_BIT,                  // Compute阶段
	COMPUTE_BINDING_NUMBER
)
```

**Shader阶段标志说明**：
- `VK_SHADER_STAGE_FRAGMENT_BIT`：片段着色器（路径追踪）
- `VK_SHADER_STAGE_COMPUTE_BIT`：计算着色器
- `VK_SHADER_STAGE_VERTEX_BIT`：顶点着色器（一般不用）

**起到的作用**：
- 告诉Vulkan哪些Shader阶段会访问该缓冲区
- 用于验证和优化

---

## 步骤6：实现Compute Shader（如果需要）

### 何时需要Compute Shader

1. **初始化数据**：每帧开始前清零或重置缓冲区
2. **预处理**：对数据进行归一化、聚合等操作
3. **后处理**：从累积数据中提取最终结果

### 文件位置
新建 `.comp` 文件，例如 `resetyourfeature.comp`、`prepareyourfeature.comp`

---

### 6.1 Reset Shader（初始化）

```glsl
#version 460
#extension GL_EXT_ray_query : require

// 引入公共头文件
#include "raycommon.glsl"

// 工作组大小（推荐64，适配大多数GPU）
layout (local_size_x = 64) in;

// 声明缓冲区（binding号与main.cpp一致）
layout(binding = COMPUTE_BINDING_NUMBER, set = 0) buffer YourBuffer
{
	YourDataStructure data[];
} yourBuffer;

void main()
{
	uint index = gl_GlobalInvocationID.x;
	
	// 边界检查
	if (index >= ELEMENT_COUNT)
		return;
	
	// 初始化为默认值
	yourBuffer.data[index].field1 = INIT_VALUE_1;
	yourBuffer.data[index].field2 = INIT_VALUE_2;
	// ...
}
```

**为什么要添加**：
- 每帧开始前需要清除上一帧的数据
- 并行初始化比串行快得多

**工作组大小选择**：
- 64是常用值，适配大多数GPU的warp/wavefront大小
- AMD：wavefront = 64
- NVIDIA：warp = 32（但64也能高效运行）
- 可以根据GPU优化调整（32、64、128、256）

**Dispatch计算**：
```cpp
// 元素数量向上取整到工作组大小的倍数
uint32_t groupCount = (ELEMENT_COUNT + 63) / 64;
dispatchCompute(pipeline, glm::uvec3(groupCount, 1, 1), barrierIndex);
```

**与什么搭配**：
- 与 `dispatchCompute` 调用配合
- 在渲染循环的开始阶段调用

**起到的作用**：
- 快速并行地初始化大量数据
- 为本帧的数据累积做准备

---

### 6.2 Prepare Shader（预处理）

```glsl
#version 460
#extension GL_EXT_ray_query : require

#include "raycommon.glsl"

layout (local_size_x = 64) in;

layout(binding = COMPUTE_BINDING_NUMBER, set = 0) buffer YourBuffer
{
	YourDataStructure data[];
} yourBuffer;

void main()
{
	uint index = gl_GlobalInvocationID.x;
	
	if (index >= ELEMENT_COUNT)
		return;
	
	// 归一化或其他预处理
	if (yourBuffer.data[index].count > 0)
	{
		yourBuffer.data[index].average = 
			yourBuffer.data[index].sum / float(yourBuffer.data[index].count);
	}
	else
	{
		yourBuffer.data[index].average = 0.0;
	}
}
```

**为什么要添加**：
- 路径追踪阶段累积原始数据
- 使用前需要归一化或转换

**执行时机**：
- 路径追踪完成后
- 下一帧使用这些数据前

**与什么搭配**：
- 在渲染循环中，reset之后、路径追踪之后调用
- 与路径追踪Shader的数据累积配合

---

## 步骤7：创建和管理Pipeline

### 文件位置
`main.cpp`

### 需要添加的内容

#### 7.1 声明Pipeline

```cpp
struct {
	// ... 其他pipelines
	VkPipeline resetYourFeature;      // 新增
	VkPipeline prepareYourFeature;    // 新增
} compute.pipelines;
```

**为什么要添加**：
- Pipeline包含编译后的Shader和配置
- 需要在命令缓冲区中绑定后才能执行

---

#### 7.2 创建Pipeline

在 `prepareComputePipelines` 或类似函数中：

```cpp
createComputePipeline(compute.pipelines.resetYourFeature, "resetyourfeature");
createComputePipeline(compute.pipelines.prepareYourFeature, "prepareyourfeature");
```

**命名规则**：
- 文件名：`resetyourfeature.comp`、`prepareyourfeature.comp`
- Pipeline名：与文件名对应（不含扩展名）
- 编译后的SPIR-V：`resetyourfeature.comp.spv`

**与什么搭配**：
- 与Shader文件（`.comp`）配合
- 与描述符集布局配合

---

#### 7.3 销毁Pipeline

在清理函数中：

```cpp
vkDestroyPipeline(device, compute.pipelines.resetYourFeature, nullptr);
vkDestroyPipeline(device, compute.pipelines.prepareYourFeature, nullptr);
```

**注意事项**：
- 必须在 `vkDestroyDevice` 之前
- 建议按创建顺序的反序销毁

---

#### 7.4 调度Compute Shader

在渲染循环的合适位置：

```cpp
// Reset阶段（每帧开始）
if (specializationData.yourFeature == 1)
{
	uint32_t groupCount = (ELEMENT_COUNT + 63) / 64;
	dispatchCompute(
		compute.pipelines.resetYourFeature,
		glm::uvec3(groupCount, 1, 1),
		BARRIER_INDEX_FOR_RESET  // 内存屏障索引
	);
}

// ... 执行路径追踪 ...

// Prepare阶段（路径追踪后）
if (specializationData.yourFeature == 1)
{
	uint32_t groupCount = (ELEMENT_COUNT + 63) / 64;
	dispatchCompute(
		compute.pipelines.prepareYourFeature,
		glm::uvec3(groupCount, 1, 1),
		BARRIER_INDEX_FOR_PREPARE
	);
}
```

**执行顺序关键**：
1. Reset Shader：清空缓冲区
2. 内存屏障（确保reset完成）
3. 路径追踪：累积数据
4. 内存屏障（确保累积完成）
5. Prepare Shader：处理数据
6. 内存屏障（确保处理完成）
7. 下一帧使用数据

**Barrier索引说明**：
- 每个compute dispatch后需要内存屏障
- 索引用于标识不同的同步点
- 查看现有代码的barrier索引分配

---

## 步骤8：在路径追踪Shader中实现采样逻辑

### 文件位置
`pathtracing.frag`

### 需要添加的内容

#### 8.1 前向声明（如果需要）

如果新功能的函数需要调用后面定义的函数：

```glsl
// 在文件开头或函数定义前添加
void existingFunction(/* 参数列表 */);
```

**为什么要添加**：
- GLSL要求函数在使用前必须声明或定义
- 避免编译错误

**VXPG示例**：
- `sampleIntraVoxel` 需要调用 `readHit`
- 但 `readHit` 定义在后面
- 因此需要前向声明

---

#### 8.2 实现核心函数

根据功能需求，可能需要实现以下类型的函数：

##### 8.2.1 数据更新函数

```glsl
void updateYourData(vec3 position, vec3 direction, vec3 contribution)
{
	// 计算索引（例如体素索引、哈希键等）
	int index = computeIndex(position, direction);
	
	if (index >= 0 && index < ELEMENT_COUNT)
	{
		// 使用原子操作更新数据
		atomicAdd(yourBuffer.data[index].accumulator, luminance(contribution));
		atomicAdd(yourBuffer.data[index].sampleCount, 1);
	}
}
```

**为什么要添加**：
- 路径追踪过程中积累统计数据
- 为下一帧的采样提供引导信息

**原子操作注意事项**：
1. **浮点数原子操作**：
   - `atomicAdd(float)` 需要扩展支持（大多数现代GPU支持）
   - `atomicMin/Max(float)` 不支持，需转换为uint
   
2. **性能影响**：
   - 原子操作会导致线程竞争
   - 尽量减少原子操作的数量
   - 考虑使用更粗粒度的划分减少冲突

**与什么搭配**：
- 在路径追踪的关键点调用（命中光源、环境光采样等）
- 与Compute Shader的预处理配合

---

##### 8.2.2 采样函数

```glsl
bool sampleYourMethod(
	vec3 shadingPos,          // 着色点位置
	vec3 normal,              // 表面法线
	inout uint seed,          // 随机数种子
	out vec3 direction,       // 输出：采样方向
	out float pdf             // 输出：概率密度
)
{
	// 1. 根据数据选择采样策略
	int selectedIndex = selectBestCandidate(shadingPos, normal, seed);
	
	if (selectedIndex < 0)
		return false;  // 失败，fallback到BSDF
	
	// 2. 根据选择结果生成方向
	direction = generateDirection(selectedIndex, seed);
	
	// 3. 计算PDF
	pdf = computePDF(selectedIndex, direction);
	
	return pdf > 0.0;
}
```

**为什么要添加**：
- 根据积累的数据引导光线采样
- 提高重要方向的采样概率
- 加速收敛

**返回值设计**：
- `true`：采样成功，使用引导方法
- `false`：采样失败，fallback到BSDF采样

**与什么搭配**：
- 在路径追踪主循环中调用
- 与BSDF采样配合（MIS或fallback）

---

#### 8.3 集成到路径追踪主循环

在 `main()` 函数的路径追踪循环中：

```glsl
// 在现有的路径引导方法后添加
else if (YOUR_FEATURE == 1)
{
	// 采用新的引导方法
	vec3 guidedDirection;
	float guidedPdf;
	
	if (sampleYourMethod(hitPos, hitNormal, seed, guidedDirection, guidedPdf))
	{
		direction = guidedDirection;
		pdf = guidedPdf * ubo.probability;
		
		// 计算几何项
		nDotL = max(0.0, dot(hitNormal, direction));
		
		// MIS：混合BSDF的PDF
		if (GUIDING_MIS == 1 && nDotL > 0.0)
		{
			float bsdfPdf = evaluateBSDFPdf(hitNormal, direction, hitMaterial, ...);
			pdf += bsdfPdf * (1.0 - ubo.probability);
		}
	}
	else
	{
		// Fallback到BSDF采样
		direction = sampleBSDF(hitNormal, seed, hitMaterial, ...);
		pdf = evaluateBSDFPdf(...);
		nDotL = max(0.0, dot(hitNormal, direction));
	}
	
	// 检查有效性
	if (nDotL <= 0.0 || pdf <= 0.0)
		break;
}
```

**执行流程**：
1. 根据 `ubo.probability` 决定使用引导方法或BSDF
2. 引导方法成功：使用引导方向
3. 引导方法失败或概率决定：使用BSDF采样
4. MIS：混合两种方法的PDF

**与什么搭配**：
- 与特化常量（`YOUR_FEATURE`）配合
- 与UI的概率滑块（`ubo.probability`）配合
- 与MIS开关（`GUIDING_MIS`）配合

---

#### 8.4 添加数据更新调用

在路径追踪的关键位置调用更新函数：

```glsl
// 1. 直接光照（命中光源）
if (PATH_GUIDING == 1)
{
	if (YOUR_FEATURE == 1)
		updateYourData(hitPos, lightDir, lightContribution);
	else if (VXPG == 1)
		updateBoundingVoxel(hitPos, lightContribution);
	// ... 其他方法
}

// 2. 环境光采样
if (PATH_GUIDING == 1)
{
	if (YOUR_FEATURE == 1)
		updateYourData(hitPos, envDir, envContribution);
	// ...
}

// 3. 路径终止时的回传
if (YOUR_FEATURE == 1)
{
	vec3 incidentRadiance = terminalRadiance;
	for (int k = pathDepth; k >= 0; --k)
	{
		updateYourData(
			pathVertices[k].position,
			pathVertices[k].direction,
			incidentRadiance
		);
		incidentRadiance *= pathVertices[k].throughput;
	}
}
```

**更新时机说明**：

1. **直接光照**：
   - NEE成功命中光源
   - 记录该位置和方向的亮度

2. **环境光采样**：
   - 光线逃逸到环境贴图
   - 记录环境光的贡献

3. **路径回传**：
   - 从路径终点向起点传播辐照度
   - 更新路径上每个顶点的数据

**与什么搭配**：
- 与数据更新函数配合
- 与Compute Shader的预处理配合

---

## 步骤9：添加数据更新逻辑

### 原子操作最佳实践

#### 9.1 原子操作类型选择

```glsl
// 1. 累加器：使用 atomicAdd
atomicAdd(data[index].sum, value);
atomicAdd(data[index].count, 1);

// 2. 最小/最大值：uint类型
atomicMin(data[index].minValue, floatBitsToUint(value));  // float转uint
atomicMax(data[index].maxValue, floatBitsToUint(value));

// 3. 复杂更新：使用 atomicCompSwap（比较交换）
uint old = data[index].value;
uint expected;
do {
	expected = old;
	uint newValue = computeNewValue(uintBitsToFloat(old));
	old = atomicCompSwap(data[index].value, expected, floatBitsToUint(newValue));
} while (old != expected);
```

**原子操作性能提示**：
- 尽量减少原子操作的数量
- 批量更新比频繁单次更新更高效
- 考虑使用更粗粒度的数据结构减少冲突

---

#### 9.2 浮点数原子操作技巧

**问题**：GLSL不支持 `atomicMin/Max(float)`

**解决方案**：利用IEEE 754浮点数的位表示特性

```glsl
// 写入：float -> uint
uint valueAsUint = floatBitsToUint(myFloat);
atomicMin(data[index].minValueUint, valueAsUint);

// 读取：uint -> float
float minValue = uintBitsToFloat(data[index].minValueUint);
```

**原理说明**：
- IEEE 754浮点数格式：符号位(1) + 指数(8/11) + 尾数(23/52)
- 对于正数：指数越大，uint值越大
- 因此uint的比较结果与float相同

**注意事项**：
- 仅适用于正数（负数的符号位会导致大小关系反转）
- 初始化min为大正数（`1e30`），max为0或负数（`-1e30`）
- NaN和Inf需要特殊处理

---

## 步骤10：集成UI控制

### 文件位置
`main.cpp` - UI构建函数

### 需要添加的内容

#### 10.1 添加SpecializationData字段

```cpp
struct SpecializationData {
	// ... 其他字段
	int32_t yourFeature = false;  // 新增，默认关闭
} specializationData;
```

**为什么要添加**：
- 存储功能的开关状态
- 传递给Shader的特化常量

**类型说明**：
- 使用 `int32_t` 而不是 `bool`（Vulkan规范要求）
- `false = 0`, `true = 1`

**与什么搭配**：
- 与Shader端的 `layout (constant_id = XX) const int YOUR_FEATURE` 配对
- 与UI复选框的变量绑定

---

#### 10.2 添加Specialization Map Entry

```cpp
std::array<VkSpecializationMapEntry, N> specializationMapEntries = {  // 增加数组大小
	// ... 其他entries
	vks::initializers::specializationMapEntry(
		CONSTANT_ID,                                  // 与Shader的constant_id对应
		offsetof(SpecializationData, yourFeature),    // 字段偏移
		sizeof(SpecializationData::yourFeature)       // 字段大小
	),
};
```

**为什么要添加**：
- 建立C++变量到Shader常量的映射
- Vulkan通过这个映射传递特化常量的值

**参数说明**：
- **constantID**：与Shader中的 `constant_id` 对应
- **offset**：字段在结构体中的偏移量（用 `offsetof` 自动计算）
- **size**：字段的大小（用 `sizeof` 自动计算）

**与什么搭配**：
- 与 `SpecializationData` 结构体配合
- 与Shader的特化常量配合

**注意事项**：
- 数组大小需要增加（如从32变为33）
- `offsetof` 和 `sizeof` 保证内存布局正确

---

#### 10.3 添加UI复选框

```cpp
if (overlay->header("Path Guiding"))
{
	if (overlay->checkBox("Path Guiding", &specializationData.pathGuiding))
		preparePipelines();  // 重新编译Pipeline
	
	// ... 其他方法的复选框
	
	if (overlay->checkBox("Your Feature", &specializationData.yourFeature))  // 新增
		preparePipelines();
}
```

**为什么要添加**：
- 提供用户界面控制功能开关
- 实时切换不同的路径引导方法

**回调说明**：
- 复选框状态改变时返回 `true`
- 调用 `preparePipelines()` 重新编译Shader
- 新的特化常量值会生效

**UI布局建议**：
- 放在 "Path Guiding" 分组下
- 与其他引导方法（CDF、SSPG、VXPG等）并列
- 可以添加子选项（如概率滑块、参数调整）

---

#### 10.4 添加参数控制（可选）

如果功能需要运行时参数调整：

```cpp
if (overlay->header("Your Feature Settings"))
{
	if (overlay->sliderFloat("Parameter 1", &yourParameter1, 0.0f, 1.0f))
		updateUniformBuffer();  // 更新UBO
	
	if (overlay->sliderInt("Parameter 2", &yourParameter2, 1, 100))
		updateUniformBuffer();
	
	if (overlay->button("Reset"))
		resetYourFeature();
}
```

**参数类型**：
- `sliderFloat`：浮点数滑块
- `sliderInt`：整数滑块
- `button`：按钮（执行操作）
- `checkBox`：复选框（布尔值）

**参数传递方式**：
- **Uniform Buffer**：频繁变化的参数（每帧或按需更新）
- **Push Constants**：小型参数（< 128字节）
- **Specialization Constants**：编译时常量（需要重新编译）

---

## 步骤11：添加SpecializationData映射

### 确保特化常量正确传递

#### 11.1 检查VkSpecializationInfo

确保在创建Pipeline时传递了特化信息：

```cpp
VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(
	specializationMapEntries.size(),
	specializationMapEntries.data(),
	sizeof(specializationData),
	&specializationData
);

// 在创建Pipeline时使用
pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
```

**为什么重要**：
- 不设置 `pSpecializationInfo` 会使用常量的默认值
- 默认值在Shader中定义（通常为0）

---

#### 11.2 验证常量ID不冲突

```cpp
// 建议维护一个文档或注释列出所有常量ID
/*
 * Specialization Constant ID分配表：
 * 0-9:   基础功能
 * 10-19: 调试模式
 * 20-31: 路径引导方法
 *   20: PATH_GUIDING
 *   21: HASHING
 *   22: CDF
 *   23: SSPG
 *   24: SGM
 *   32: VXPG
 *   33: YOUR_FEATURE  // 新增
 */
```

---

## 步骤12：销毁资源

### 文件位置
`main.cpp` - 析构函数或清理函数

### 需要销毁的资源

#### 12.1 GPU缓冲区

```cpp
storageBuffers.yourBuffer.destroy();
```

**销毁顺序**：
- 建议在销毁Pipeline之前
- 必须在 `vkDestroyDevice` 之前

---

#### 12.2 Pipeline

```cpp
vkDestroyPipeline(device, compute.pipelines.resetYourFeature, nullptr);
vkDestroyPipeline(device, compute.pipelines.prepareYourFeature, nullptr);
```

**销毁顺序**：
- 可以在任何时候销毁（只要不在使用中）
- 建议按创建顺序的反序

---

#### 12.3 描述符集和布局

一般不需要显式销毁（随descriptor pool一起销毁），但如果手动管理：

```cpp
vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
// 描述符集本身通过 vkFreeDescriptorSets 或销毁pool释放
```

---

## 常见错误和解决方案

### 错误1：编译失败 - 未找到binding

**错误信息**：
```
Error: binding X is not declared in the shader
```

**原因**：
- Shader中声明了binding，但C++端没有绑定
- 或binding号不匹配

**解决方案**：
1. 检查 `writeDescriptorSet` 中的binding号与Shader一致
2. 检查 `descriptorSetLayoutBinding` 是否添加
3. 确认 `updateDescriptorSets` 被调用

---

### 错误2：运行时错误 - 访问越界

**错误信息**：
```
Validation Layer: Buffer access out of bounds
```

**原因**：
- 缓冲区大小计算错误
- Shader中的索引计算错误

**解决方案**：
1. 验证缓冲区大小：`ELEMENT_COUNT * sizeof(Structure)`
2. 在Shader中添加边界检查：`if (index >= ELEMENT_COUNT) return;`
3. 检查索引计算逻辑（如体素索引、哈希函数）

---

### 错误3：渲染结果错误 - 数据未更新

**症状**：
- 功能开启后渲染结果没有变化
- 或出现黑屏、闪烁等异常

**可能原因**：
1. **特化常量未传递**：
   - 检查 `specializationMapEntries` 是否添加
   - 检查 `VkSpecializationInfo` 是否设置

2. **内存屏障缺失**：
   - Compute Shader执行后需要屏障
   - 否则后续读取可能得到旧数据

3. **原子操作冲突**：
   - 检查是否正确使用原子操作
   - 验证数据竞争是否被处理

4. **PDF计算错误**：
   - 检查采样PDF是否正确归一化
   - MIS权重是否正确

**调试方法**：
- 使用Validation Layers检查Vulkan错误
- 添加Debug模式可视化中间数据
- 使用RenderDoc等工具检查GPU状态

---

### 错误4：性能问题 - 帧率下降

**可能原因**：
1. **缓冲区过大**：
   - 减少体素数量或降低分辨率
   - 使用压缩格式（16位浮点）

2. **原子操作过多**：
   - 减少原子操作的数量
   - 使用更粗粒度的划分

3. **Compute Shader未优化**：
   - 调整工作组大小（64, 128, 256）
   - 检查是否有warp/wavefront分歧

4. **内存带宽瓶颈**：
   - 使用纹理替代buffer（利用缓存）
   - 合并访问模式

---

## 完整的功能添加检查清单

在实现新功能时，使用此清单逐项检查：

### Shader端
- [ ] 在 `raycommon.glsl` 中添加特化常量（`layout (constant_id = XX)`）
- [ ] 定义数据结构（考虑内存对齐和原子操作）
- [ ] 在 `pathtracing.frag` 中声明buffer binding
- [ ] 在 `.comp` 文件中声明buffer binding（如需要）
- [ ] 实现Reset Shader（初始化）
- [ ] 实现Prepare Shader（预处理）
- [ ] 实现数据更新函数（`updateYourData`）
- [ ] 实现采样函数（`sampleYourMethod`）
- [ ] 在路径追踪主循环中集成
- [ ] 在关键位置添加数据更新调用
- [ ] 添加必要的前向声明

### C++端
- [ ] 在 `main.cpp` 中定义对应的数据结构
- [ ] 在 `storageBuffers` 中添加buffer成员
- [ ] 调用 `createBuffer` 创建GPU缓冲区
- [ ] 在Ray Tracing Pass中添加 `writeDescriptorSet`
- [ ] 在Compute Pass中添加 `writeDescriptorSet`（如需要）
- [ ] 在Ray Tracing Pipeline布局中添加 `descriptorSetLayoutBinding`
- [ ] 在Compute Pipeline布局中添加 `descriptorSetLayoutBinding`（如需要）
- [ ] 创建Compute Pipeline（reset和prepare）
- [ ] 在渲染循环中调度Compute Shader
- [ ] 在 `SpecializationData` 中添加字段
- [ ] 在 `specializationMapEntries` 中添加映射
- [ ] 在UI中添加复选框
- [ ] 添加参数控制（可选）
- [ ] 销毁buffer（在清理函数中）
- [ ] 销毁pipeline（在清理函数中）

### 验证和测试
- [ ] 检查Vulkan Validation Layers输出（无错误/警告）
- [ ] 验证功能开关有效（UI复选框）
- [ ] 测试与其他方法的互斥性（不能同时开启多个）
- [ ] 测试MIS模式（如适用）
- [ ] 性能测试（帧率、GPU占用）
- [ ] 渲染结果验证（与参考结果对比）
- [ ] 内存泄漏检查

---

## 高级话题

### 多缓冲策略

对于需要跨帧累积的数据，考虑使用双缓冲或多缓冲：

```cpp
struct {
	vks::Buffer current;   // 当前帧使用
	vks::Buffer previous;  // 上一帧数据
} yourBuffers;

// 每帧交换
void swapBuffers()
{
	std::swap(yourBuffers.current, yourBuffers.previous);
}
```

**优点**：
- 避免读写冲突
- 支持时序滤波和平滑

---

### 自适应参数调整

根据场景复杂度或性能自动调整参数：

```cpp
// 根据帧率调整质量
if (frameTime > targetFrameTime)
{
	// 降低质量（如减少体素分辨率）
	pushConstants.gridDim /= 2;
}
else if (frameTime < targetFrameTime * 0.8)
{
	// 提高质量
	pushConstants.gridDim *= 2;
}
```

---

### 调试可视化

添加Debug模式可视化中间结果：

```glsl
if (DEBUG == YOUR_DEBUG_MODE)
{
	// 可视化数据（如体素亮度、采样密度等）
	int gridIndex = getGridIndex(hitPos);
	if (gridIndex >= 0)
	{
		float value = yourBuffer.data[gridIndex].someField;
		outColor = vec3(value);  // 灰度图
		// 或使用颜色映射
		outColor = heatmapColor(value, minValue, maxValue);
	}
	return;
}
```

在UI中添加Debug模式选择：

```cpp
if (overlay->comboBox("Debug Mode", &debugMode, debugModeNames))
	reset();
```

---

## 总结

添加新的路径引导功能需要在多个层次进行协调修改：

1. **数据层**：定义数据结构和缓冲区（Shader + C++）
2. **计算层**：实现数据的初始化、更新和预处理（Compute Shader）
3. **渲染层**：实现采样和引导逻辑（Ray Tracing Shader）
4. **接口层**：提供UI控制和参数调整（C++ UI）
5. **管理层**：正确创建和销毁资源（C++ 资源管理）

关键原则：
- **一致性**：Shader和C++的结构体、binding号、常量ID必须匹配
- **完整性**：创建的资源必须正确销毁，添加的binding必须有布局
- **顺序性**：注意执行顺序和内存屏障，避免数据竞争
- **验证性**：使用Validation Layers和边界检查，及早发现错误

通过系统性地遵循这些步骤，可以确保新功能的实现完整、正确且高效。
