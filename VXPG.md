# VXPG (Voxel-based Path Guiding) 实现说明

## 概述

本文档详细说明了在 VulkanPG 项目中实现 VXPG（基于体素的路径引导）方法的所有代码修改。VXPG 是一种实时路径引导技术，通过在场景中构建"亮度体素地图"来指导光线采样，从而加速路径追踪的收敛速度。

## 理论基础

VXPG 的核心思想：
1. **构建亮度地图**：将场景划分为体素网格，记录每个体素的辐照度和几何边界盒（AABB）
2. **体素选择**：根据体素的亮度、可见性和 BSDF 匹配度加权选择候选体素
3. **体素内采样**：在选中体素的 AABB 内均匀采样，然后追踪到实际几何表面
4. **无偏性**：采样概率 = P(选择体素) × P(体素内采样点)，保证渲染结果无偏

---

## 一、数据结构定义

### 1.1 Shader 端结构体 (`raycommon.glsl`)

#### 添加 VXPG 常量

```glsl
layout (constant_id = 32) const int VXPG = 0;
```

**理由**：
- 使用 specialization constant 实现运行时开关，避免重新编译 shader
- 与其他路径引导方法（CDF, HASHING, SSPG, SGM）保持一致的启用方式

#### 添加 BoundingVoxel 结构体

```glsl
struct BoundingVoxel
{
	uint aabbMinX;
	uint aabbMinY;
	uint aabbMinZ;
	float totalIrradiance;
	uint aabbMaxX;
	uint aabbMaxY;
	uint aabbMaxZ;
	uint sampleCount;
};
```

**设计说明**：
- **使用 uint 存储 AABB 坐标**：因为 GLSL 不支持对浮点数直接使用 `atomicMin/Max`，需要通过 `floatBitsToUint` 转换后使用原子操作
- **IEEE 754 特性**：正浮点数转为 uint 后保持大小关系不变，可以直接比较
- **totalIrradiance**：累积的辐照度，用于计算体素权重
- **sampleCount**：采样数量，用于归一化辐照度

### 1.2 C++ 端结构体 (`main.cpp`)

```cpp
/*
用于存储每个体素的几何边界和光照信息
*/
struct BoundingVoxel
{
	uint32_t aabbMinX;     // 最小角
	uint32_t aabbMinY;
	uint32_t aabbMinZ;
	float totalIrradiance; // 累积的辐照度
	uint32_t aabbMaxX;
	uint32_t aabbMaxY;
	uint32_t aabbMaxZ;
	uint32_t sampleCount;  // 采样次数
};
```

**理由**：
- 必须与 shader 端结构体内存布局完全一致
- 用于 buffer 大小计算和可能的调试读取

---

## 二、存储缓冲区和描述符

### 2.1 添加 boundingVoxels 缓冲区 (`main.cpp`)

#### 在 storageBuffers 中添加

```cpp
struct {
	// ... 其他 buffers
	vks::Buffer boundingVoxels;  // 新增
	// ...
} storageBuffers;
```

#### 创建缓冲区

```cpp
createBuffer(storageBuffers.boundingVoxels, 
	pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z * sizeof(BoundingVoxel), 
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
```

**参数说明**：
- **大小**：体素总数 × 结构体大小
- **用途标志**：`STORAGE_BUFFER` 用于 shader 读写
- **内存属性**：`DEVICE_LOCAL` 放在 GPU 显存，性能最优

#### 销毁缓冲区

```cpp
storageBuffers.boundingVoxels.destroy();
```

### 2.2 绑定到描述符集

#### Ray Tracing Pass (binding = 28)

```cpp
vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, 
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 28, 
	&storageBuffers.boundingVoxels.descriptor)
```

**binding 28**：在 `pathtracing.frag` 中访问，用于更新和采样

#### Compute Pass (binding = 7)

```cpp
vks::initializers::writeDescriptorSet(compute.descriptorSet, 
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, 
	&storageBuffers.boundingVoxels.descriptor)
```

**binding 7**：在 compute shader 中访问，用于重置和准备

#### 添加描述符集布局

```cpp
// Ray Tracing
vks::initializers::descriptorSetLayoutBinding(
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 28)

// Compute
vks::initializers::descriptorSetLayoutBinding(
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7)
```

---

## 三、Compute Shader 实现

### 3.1 Reset Shader (`resetvxpg.comp`)

```glsl
void main()
{
	uint voxelIndex = gl_GlobalInvocationID.x;
	if (voxelIndex >= gridDim.x * gridDim.y * gridDim.z)
		return;

	// 初始化为极值，准备做 min/max
	boundingVoxels[voxelIndex].aabbMinX = floatBitsToUint(1e30);
	boundingVoxels[voxelIndex].aabbMinY = floatBitsToUint(1e30);
	boundingVoxels[voxelIndex].aabbMinZ = floatBitsToUint(1e30);
	boundingVoxels[voxelIndex].aabbMaxX = floatBitsToUint(-1e30);
	boundingVoxels[voxelIndex].aabbMaxY = floatBitsToUint(-1e30);
	boundingVoxels[voxelIndex].aabbMaxZ = floatBitsToUint(-1e30);
	boundingVoxels[voxelIndex].totalIrradiance = 0.0;
	boundingVoxels[voxelIndex].sampleCount = 0;
}
```

**执行时机**：每帧开始前调用

**工作组大小**：`local_size_x = 64`

**Dispatch 配置**：
```cpp
glm::uvec3((gridDim.x * gridDim.y * gridDim.z + 63u) / 64, 1, 1)
```

**逻辑说明**：
- 每个线程处理一个体素
- AABB min 初始化为最大值，max 初始化为最小值，便于后续 atomic min/max 更新
- 辐照度和采样计数归零

### 3.2 Prepare Shader (`preparevxpg.comp`)

```glsl
void main()
{
	uint voxelIndex = gl_GlobalInvocationID.x;
	if (voxelIndex >= gridDim.x * gridDim.y * gridDim.z)
		return;

	// 归一化辐照度
	if (boundingVoxels[voxelIndex].sampleCount > 0)
	{
		boundingVoxels[voxelIndex].totalIrradiance /= 
			float(boundingVoxels[voxelIndex].sampleCount);
	}
	else
	{
		boundingVoxels[voxelIndex].totalIrradiance = 0.0;
	}
}
```

**执行时机**：路径追踪后，使用前

**逻辑说明**：
- 将累积的辐照度除以采样数，得到平均辐照度
- 用于体素选择时的权重计算

### 3.3 主程序集成 (`main.cpp`)

#### 添加 Pipeline

```cpp
struct {
	VkPipeline reset;
	VkPipeline prepare;
	VkPipeline resetVXPG;    // 新增
	VkPipeline prepareVXPG;  // 新增
} pipelines;
```

#### 创建 Pipeline

```cpp
createComputePipeline(compute.pipelines.resetVXPG, "resetvxpg");
createComputePipeline(compute.pipelines.prepareVXPG, "preparevxpg");
```

#### 销毁 Pipeline

```cpp
vkDestroyPipeline(device, compute.pipelines.resetVXPG, nullptr);
vkDestroyPipeline(device, compute.pipelines.prepareVXPG, nullptr);
```

#### Reset 调度

```cpp
if (specializationData.vxpg == 1)
{
	dispatchCompute(compute.pipelines.resetVXPG, 
		glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 63u) / 64, 1, 1), 
		6);
}
```

#### Prepare 调度

```cpp
if (specializationData.vxpg == 1)
{
	dispatchCompute(compute.pipelines.prepareVXPG, 
		glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 63u) / 64, 1, 1), 
		0);
}
```

---

## 四、路径追踪 Shader 实现 (`pathtracing.frag`)

### 4.1 前向声明

```glsl
void readHit(rayQueryEXT rayQuery, inout vec3 hitPos, inout vec3 hitNormal, 
	inout HitMaterial hitMaterial, vec3 dir);
```

**理由**：
- `sampleIntraVoxel` 需要调用 `readHit`，但 `readHit` 定义在后面
- GLSL 要求函数在调用前必须声明或定义

### 4.2 更新边界体素

```glsl
void updateBoundingVoxel(vec3 position, vec3 irradiance)
{
    // position是光线命中点的世界坐标
    // irradiance是该命中点接收到的辐照度
	int gridIndex = getGridIndex(position);
	if (gridIndex >= 0)
	{
		// 转换为 uint 进行原子比较
		uint posX = floatBitsToUint(position.x);
		uint posY = floatBitsToUint(position.y);
		uint posZ = floatBitsToUint(position.z);
		
		// 原子更新 AABB
		atomicMin(boundingVoxels[gridIndex].aabbMinX, posX);
		atomicMin(boundingVoxels[gridIndex].aabbMinY, posY);
		atomicMin(boundingVoxels[gridIndex].aabbMinZ, posZ);
		atomicMax(boundingVoxels[gridIndex].aabbMaxX, posX);
		atomicMax(boundingVoxels[gridIndex].aabbMaxY, posY);
		atomicMax(boundingVoxels[gridIndex].aabbMaxZ, posZ);
		
		// 累积辐照度
		atomicAdd(boundingVoxels[gridIndex].totalIrradiance, luminance(irradiance));
		atomicAdd(boundingVoxels[gridIndex].sampleCount, 1);
	}
}
```

**调用时机**：
1. **直接光照**：聚光灯、环境光 NEE 命中时
2. **环境光**：光线逃逸到环境贴图时
3. **路径顶点回传**：从终点向起点传播辐照度

**原子操作原理**：
- IEEE 754 浮点数的位表示：符号位 + 指数 + 尾数
- 对于正数：指数越大，uint 值越大
- 因此 `atomicMin/Max(uint)` 可以正确比较浮点大小

### 4.3 体素选择

```glsl
// 这个函数的目的是从所有体素中，根据"贡献度"加权选择一个最有可能包含光源的体素
int sampleVoxel(vec3 shadingPos, vec3 normal, inout uint seed, out float pdf)
{
	const int numSamples = 8; // 采样 8 个候选体素
	float weights[numSamples];
	int indices[numSamples];
	float totalWeight = 0.0;
	
	// 收集候选体素
	for (int i = 0; i < numSamples; ++i)
	{
		// 随机选择体素（可优化为空间聚类）
		int voxelIdx = int(rnd(seed) * float(gridDim.x * gridDim.y * gridDim.z));
		indices[i] = voxelIdx;
		
		if (boundingVoxels[voxelIdx].sampleCount == 0)
		{
			weights[i] = 0.0;
			continue;
		}
		
		// 计算体素中心
		int x = voxelIdx % gridDim.x;
		int y = (voxelIdx / gridDim.x) % gridDim.y;
		int z = voxelIdx / (gridDim.x * gridDim.y);
		vec3 voxelCenter = gridBegin + (vec3(x, y, z) + 0.5) * cellSize;
		
		// 估算贡献：辐照度 × 可见性 × 几何项
		vec3 toVoxel = voxelCenter - shadingPos;
		float dist = length(toVoxel);
		vec3 dir = toVoxel / max(dist, 0.001);
		
		// 简单可见性检查（朝向正面）
		float visibility = max(0.0, dot(normal, dir));
		
		// 权重：辐照度 × 可见性 / 距离平方
		float irradiance = boundingVoxels[voxelIdx].totalIrradiance;
		weights[i] = irradiance * visibility / max(dist * dist, 0.01);
		totalWeight += weights[i];
	}
	
	if (totalWeight < 1e-6)
	{
		pdf = 0.0;
		return -1;
	}
	
	// 按权重采样体素
	float r = rnd(seed) * totalWeight;
	float cumulative = 0.0;
	for (int i = 0; i < numSamples; ++i)
	{
		cumulative += weights[i];
		if (r <= cumulative)
		{
			pdf = weights[i] / totalWeight;
			return indices[i];
		}
	}
	
	pdf = weights[numSamples - 1] / totalWeight;
	return indices[numSamples - 1];
}
```

**采样策略**：
1. **候选数量**：8 个（平衡性能和质量）
2. **选择方法**：随机选择（实际应用可用空间哈希优化）
3. **权重计算**：
   - 辐照度：体素的平均亮度
   - 可见性：法线与方向的点积（简化版）
   - 几何项：1/距离²（考虑立体角衰减）

### 4.4 体素内采样

```glsl
bool sampleIntraVoxel(int voxelIdx, inout uint seed, 
	out vec3 sampledPos, out vec3 sampledNormal, out HitMaterial sampledMaterial)
{
	// 读取 AABB（uint 转 float）
	vec3 aabbMin = vec3(
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinX),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinY),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinZ)
	);
	vec3 aabbMax = vec3(
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxX),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxY),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxZ)
	);
	
	// 检查 AABB 是否有效
	if (any(greaterThan(aabbMin, aabbMax)))
		return false;
	
	// 在 AABB 内均匀采样点
	vec3 samplePoint = aabbMin + vec3(rnd(seed), rnd(seed), rnd(seed)) * (aabbMax - aabbMin);
	
	// 从 AABB 中心向采样点追踪
	vec3 aabbCenter = (aabbMin + aabbMax) * 0.5;
	vec3 toSample = samplePoint - aabbCenter;
	float rayDist = length(toSample);
	vec3 rayDir = toSample / max(rayDist, 0.001);
	
	// 光线追踪
	rayQueryEXT rayQuery;
	rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 
		aabbCenter, 0.001, rayDir, rayDist);
	while(rayQueryProceedEXT(rayQuery)) {}
	
	if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
	{
		// 使用临时变量（readHit 需要 inout，但参数是 out）
		vec3 tempPos = vec3(0.0);
		vec3 tempNormal = vec3(0.0, 0.0, 1.0);
		HitMaterial tempMaterial;
		tempMaterial.baseColor = vec3(1.0);
		tempMaterial.metallic = 0.0;
		tempMaterial.specular = 0.5;
		tempMaterial.roughness = 1.0;
		tempMaterial.metallicRoughness = true;
		tempMaterial.specularColor = vec3(0.0);
		
		readHit(rayQuery, tempPos, tempNormal, tempMaterial, rayDir);
		
		// 赋值给输出参数
		sampledPos = tempPos;
		sampledNormal = tempNormal;
		sampledMaterial = tempMaterial;
		return true;
	}
	
	return false;
}
```

**采样流程**：
1. 在 AABB 内均匀随机采样一个点
2. 从 AABB 中心向该点发射光线
3. 如果命中几何体，返回命中信息
4. 如果未命中，返回 false（fallback 到 BSDF 采样）

**无偏性保证**：
- AABB 内均匀采样的 PDF = 1 / AABB体积
- 最终 PDF = P(体素) × P(AABB内)

### 4.5 路径引导主逻辑

```glsl
else if (VXPG == 1)
{
	// VXPG 路径引导
	float voxelPdf;
	int selectedVoxel = sampleVoxel(hitPos, hitNormal, seed, voxelPdf);
	
	if (selectedVoxel >= 0 && voxelPdf > 0.0)
	{
		vec3 sampledPos, sampledNormal;
		HitMaterial sampledMaterial;
		
		// 在体素内采样
		if (sampleIntraVoxel(selectedVoxel, seed, sampledPos, sampledNormal, sampledMaterial))
		{
			direction = normalize(sampledPos - hitPos);
			nDotL = max(0.0, dot(hitNormal, direction));
			
			if (nDotL > 0.0)
			{
				// 计算 PDF
				vec3 aabbMin = vec3(/* 读取 AABB */);
				vec3 aabbMax = vec3(/* 读取 AABB */);
				float aabbVolume = max(1e-6, 
					(aabbMax.x - aabbMin.x) * (aabbMax.y - aabbMin.y) * (aabbMax.z - aabbMin.z));
				float intraVoxelPdf = 1.0 / aabbVolume;
				
				pdf = voxelPdf * intraVoxelPdf * ubo.probability;
			}
			else
			{
				// fallback 到 BSDF
			}
		}
		else
		{
			// fallback 到 BSDF
		}
	}
	else
	{
		// fallback 到 BSDF
	}
	
	if (nDotL <= 0.0 || pdf <= 0.0)
		break;
	
	if (GUIDING_MIS == 1)
		pdf += pdfBRDF(...) * (1.0 - ubo.probability);
}
```

**执行流程**：
1. 按概率 `ubo.probability` 决定使用 VXPG 或 BSDF
2. VXPG 采样：选择体素 → 体素内采样
3. 失败时 fallback 到 BSDF 采样
4. MIS：混合 VXPG 和 BSDF 的 PDF

### 4.6 辐照度更新集成

#### 直接光照（聚光灯）

```glsl
if (PATH_GUIDING == 1)
{
	if (gridIndex >= 0)
	{
		if (VXPG == 1)
			updateBoundingVoxel(hitPos, vec3(attenuation));
		else
			updateRadianceField(hitPos, lightDir, vec3(attenuation));
	}
}
```

#### 环境光 NEE

```glsl
if (PATH_GUIDING == 1)
{
	if (gridIndex >= 0)
	{
		if (VXPG == 1)
			updateBoundingVoxel(hitPos, envMapRadiance);
		else
			updateRadianceField(hitPos, neeDir, envMapRadiance);
	}
}
```

#### 路径顶点回传

```glsl
if (VXPG == 1)
{
	// VXPG：更新边界体素
	vec3 incidentRadiance = envMapRadiance;
	for (int k = j; k >= 0; --k)
	{
		updateBoundingVoxel(pathVertices[k].position, incidentRadiance);
		incidentRadiance *= pathVertices[k].throughput;
	}
}
else
{
	// 原方法：更新方向场
	vec3 incidentRadiance = envMapRadiance;
	for (int k = j; k >= 0; --k)
	{
		updateRadianceField(pathVertices[k].position, pathVertices[k].direction, incidentRadiance);
		incidentRadiance *= pathVertices[k].throughput;
	}
}
```

---

## 五、用户界面集成

### 5.1 添加 SpecializationData 字段 (`main.cpp`)

```cpp
struct SpecializationData {
	// ... 其他字段
	int32_t vxpg = false;  // 新增
} specializationData;
```

### 5.2 添加 Specialization Map Entry

```cpp
std::array<VkSpecializationMapEntry, 33> specializationMapEntries = {  // 从 32 改为 33
	// ... 其他 entries
	vks::initializers::specializationMapEntry(32, 
		offsetof(SpecializationData, vxpg), 
		sizeof(SpecializationData::vxpg)),
};
```

### 5.3 添加 UI 复选框

```cpp
if (overlay->header("Path Guiding"))
{
	if (overlay->checkBox("Path Guiding", &specializationData.pathGuiding))
		preparePipelines();
	if (overlay->button("Reset"))
		reset();
	if (overlay->checkBox("Hashing", &specializationData.hashing))
		preparePipelines();
	if (overlay->checkBox("CDF", &specializationData.cdf))
		preparePipelines();
	if (overlay->checkBox("SSPG", &specializationData.sspg))
		preparePipelines();
	if (overlay->checkBox("SGM", &specializationData.sgm))
		preparePipelines();
	if (overlay->checkBox("VXPG", &specializationData.vxpg))  // 新增
		preparePipelines();
	// ...
}
```

**UI 位置**：与 Hashing、CDF、SSPG、SGM 同级，在 Path Guiding 分组下

---

## 六、关键技术问题解决

### 6.1 原子操作浮点数

**问题**：GLSL 不支持 `atomicMin/Max(float)`

**解决方案**：
1. 使用 `uint` 存储浮点数的位表示
2. 利用 IEEE 754 特性：正数的 uint 值保持大小关系
3. 写入：`atomicMin(aabbMinX, floatBitsToUint(position.x))`
4. 读取：`uintBitsToFloat(aabbMinX)`

### 6.2 函数参数修饰符不匹配

**问题**：`readHit` 需要 `inout` 参数，但 `sampleIntraVoxel` 的输出是 `out`

**解决方案**：
```glsl
// 使用临时变量作为中介
vec3 tempPos = vec3(0.0);
vec3 tempNormal = vec3(0.0, 0.0, 1.0);
HitMaterial tempMaterial;
// ... 初始化 tempMaterial

readHit(rayQuery, tempPos, tempNormal, tempMaterial, rayDir);

// 赋值给 out 参数
sampledPos = tempPos;
sampledNormal = tempNormal;
sampledMaterial = tempMaterial;
```

### 6.3 函数前向声明

**问题**：`sampleIntraVoxel` 在 `readHit` 之前定义，但需要调用它

**解决方案**：
```glsl
// 在 VXPG 函数前添加前向声明
void readHit(rayQueryEXT rayQuery, inout vec3 hitPos, inout vec3 hitNormal, 
	inout HitMaterial hitMaterial, vec3 dir);
```

---

## 七、性能优化建议

### 7.1 体素选择优化

**当前实现**：随机选择 8 个体素

**优化方向**：
- 使用空间哈希（Spatial Hashing）预选择邻近体素
- 使用超级像素聚类（Super-pixel Clustering）
- 实现方向四叉树（Directional Quadtree）

### 7.2 AABB 采样优化

**当前实现**：AABB 中心向随机点追踪

**优化方向**：
- 直接在 AABB 表面采样（减少光线追踪次数）
- 使用重要性采样（根据表面积分布）
- 多次尝试采样（提高命中率）

### 7.3 内存访问优化

**当前实现**：每个体素 8 个字段，32 字节

**优化方向**：
- 使用 16 位浮点数存储（减少带宽）
- 分离冷热数据（常访问的和不常访问的）
- 使用纹理存储代替 buffer（利用缓存）

---

## 八、调试和验证

### 8.1 可视化调试

建议添加以下调试模式：

```glsl
if (DEBUG == 10) // VXPG Debug
{
	// 可视化体素 AABB
	outColor = vec3(aabbMin.x, aabbMin.y, aabbMin.z);
}
if (DEBUG == 11)
{
	// 可视化体素辐照度
	outColor = vec3(boundingVoxels[gridIndex].totalIrradiance);
}
if (DEBUG == 12)
{
	// 可视化采样成功率
	outColor = vec3(sampleCount / maxSamples);
}
```

### 8.2 性能监控

在 `main.cpp` 中添加时间戳查询：

```cpp
computeTimestamps[6] // resetVXPG
computeTimestamps[7] // prepareVXPG
```

---

## 九、使用方法

### 9.1 启用 VXPG

1. 勾选 "Path Guiding" 复选框
2. 勾选 "VXPG" 复选框
3. 调整 "Probability" 滑块（0.0 - 1.0）
   - 0.0：完全使用 BSDF 采样
   - 1.0：完全使用 VXPG
   - 0.5：混合采样（推荐）

### 9.2 参数调优

- **体素网格分辨率**：`pushConstants.cellSize`（在代码中硬编码为 0.5）
  - 较小：更精细，内存占用大
  - 较大：更粗糙，性能更好

- **采样概率**：`ubo.probability`
  - 简单场景：0.3 - 0.5
  - 复杂光照：0.5 - 0.8

- **MIS 模式**：`Guiding MIS` 复选框
  - 开启：混合 VXPG 和 BSDF 的 PDF（更鲁棒）
  - 关闭：纯 VXPG PDF（可能有偏差）

---

## 十、与其他方法对比

| 方法 | 存储 | 采样速度 | 收敛速度 | 内存占用 |
|------|------|----------|----------|----------|
| **VXPG** | AABB + 辐照度 | 快（简单光线追踪） | 中等 | 中等（每体素 32 字节） |
| **CDF** | 方向 CDF | 中等（二分查找） | 快 | 大（每体素 256 字节） |
| **SSPG** | 屏幕空间 GMM | 快（像素着色器） | 快 | 小（屏幕分辨率相关） |
| **Hashing** | 哈希表 | 快（哈希查找） | 中等 | 中等 |

---

## 十一、已知限制

1. **体素选择随机性**：当前使用完全随机选择，未利用空间相关性
2. **AABB 采样效率**：从中心追踪可能多次失败
3. **单次采样**：每个着色点只采样一个体素
4. **无自适应网格**：体素大小固定，未根据场景复杂度调整

---

## 十二、未来改进方向

1. **多级网格**：使用八叉树或级联网格
2. **时序重用**：跨帧复用体素信息
3. **光子映射集成**：结合 photon mapping 预计算
4. **机器学习优化**：使用神经网络预测重要体素

---

## 总结

VXPG 实现完整集成了从数据结构、compute shader、路径追踪到 UI 的所有组件。核心思想是通过构建体素化的辐照度场来加速路径采样，同时保持渲染结果的无偏性。该实现为实时路径引导提供了一种高效的替代方案，特别适合具有明显局部亮度聚集的场景。
