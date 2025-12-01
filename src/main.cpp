/*
* Vulkan Example - Screen space ambient occlusion example
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanRaytracingSample.h"
#include "VulkanglTFModel.h"
#include "NRDWrapper.h"
#include <stb_image.h>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/opencv.hpp>

constexpr int numTimestamps = 10;
constexpr int numComputeTimestamps = 10;
constexpr int incidentRadianceMapSize = 8;

class VulkanExample : public VulkanRaytracingSample
{
public:
	struct PushConstants {
		int modelIndex;
		glm::vec3 gridBegin;
		glm::ivec3 gridDim;
		float cellSize = 1.0f;
		int lobeCount = 4;
		float exponentialFactor = 0.7f;
	} pushConstants;

	float timeStampPeriod = 1e-6f;

	// Host data to take specialization constants from
	struct SpecializationData {
		int32_t textured = true;
		int32_t bumped = true;
		int32_t orthogonalize = true;
		int32_t gammaCorrection = true;
		int32_t alphaTest = true;
		int32_t toneMapping = false;
		int32_t animateNoise = true;
		int32_t nee = false;
		int32_t restirDI = false;
		int32_t envMapIS = false;
		int32_t visibilityReuse = true;
		int32_t temporalReuseDI = true;
		int32_t temporalReuseGI = true;
		int32_t spatialReuseDI = true;
		int32_t spatialReuseGI = true;
		int32_t restirGI = false;
		int32_t debug = 0;
		int32_t spotLight = false;
		int32_t environmentMap = true;
		int32_t biasCorrectionDI = true;
		int32_t biasCorrectionGI = true;
		int32_t geometricSimilarityDI = true;
		int32_t geometricSimilarityGI = true;
		int32_t jacobian = true;
		uint32_t incidentRadianceMapSize;
		int32_t pathGuiding = false;
		int32_t neeMIS = true;
		int32_t guidingMIS = true;
		int32_t hashing = false;
		int32_t cdf = false;
		int32_t sspg = false;
		int32_t sgm = true;
		int32_t vxpg = false;
	} specializationData;

	struct {
		vks::Buffer uniformBuffer;					// Uniform buffer object containing scene data
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		std::array<VkCommandBuffer, 3> commandBuffers;
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
		VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		VkDescriptorSet descriptorSet;				// Compute shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		struct
		{
			VkPipeline reset;
			VkPipeline prepare;
			VkPipeline resetVXPG;
			VkPipeline prepareVXPG;
		} pipelines;
		struct UBOCompute {							// Compute shader uniform block object
		} ubo;
	} compute;

	int32_t debugDisplayTarget = 0;
	float currentTime = 0.0f;
	struct {
		vks::Texture2D envMap;
		vks::Texture2D envHDRCache;
		vks::Texture2D whiteNoiseMap;
		vks::Texture2D blueNoiseMap;
	} textures;

	struct {
		std::vector<vkglTF::Model> models;
	} models;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 previousProjection;
		glm::mat4 previousView;
		float nearPlane = 0.1f;
		float farPlane = 64.0f;
		float alphaCutoff = 0.5f;
		glm::vec3 probePos;
		glm::vec3 viewPos;
	} uboGBuffer;

	struct {
		glm::mat4 inverseMVP;
		glm::mat4 projection;
		glm::vec3 viewPos;
		int32_t debugDisplayTarget = 0;
		int historyLength = 1;
		float exposure = 1.0f;
		float alphaCutoff = 0.5f;
		int frameNumber = 0;
		int sampleCount = 8;
		glm::mat3 envRot;
		int numBounces = 3;
		float probability = 0.5f;
		int numCandidates = 32;
		int temporalSamplesDI = 20;
		int temporalSamplesGI = 20;
		int32_t reference = false;
		glm::vec3 spotDir;
		glm::vec3 spotLightPos;
		float spotAngle = 45.0f;
		float spotLightIntensity = 1000.0f;
		int spatialReuseRadiusDI = 32;
		int spatialReuseRadiusGI = 32;
		int spatialSamplesDI = 3;
		int spatialSamplesGI = 3;
		int32_t russianRoulette = true;
		float clampValue = 100.0f;
		int32_t screenshot = false;
	} uboComposition;

	struct {
		int32_t blur = true;
		int radius = 2;
		//float depthSharpness = 10.0f;
		//float normalSharpness = 1000.0f;
		float depthSharpness = 0.003f;
		float normalSharpness = 32.0f;
	} uboBlur;
	
	// NRD Denoiser
	NRDWrapper nrdDenoiser;
	bool useNRD = false;  // UI控制开关
	uint32_t nrdFrameIndex = 0;  // NRD专用的连续frameIndex

	VkSemaphore semaphore;

	struct {
		VkPipeline gBuffer;
		VkPipeline rayTracing;
		VkPipeline blur;
		VkPipeline composition;
	} pipelines;

	struct {
		VkPipelineLayout gBuffer;
		VkPipelineLayout rayTracing;
		VkPipelineLayout blur;
		VkPipelineLayout composition;
	} pipelineLayouts;

	struct {
		const uint32_t count = 5;
		VkDescriptorSet model;
		VkDescriptorSet rayTracing;
		VkDescriptorSet blur;
		VkDescriptorSet composition;
	} descriptorSets;

	struct {
		VkDescriptorSetLayout gBuffer;
		VkDescriptorSetLayout rayTracing;
		VkDescriptorSetLayout blur;
		VkDescriptorSetLayout composition;
	} descriptorSetLayouts;

	struct {
		vks::Buffer gBuffer;
		vks::Buffer composition;
		vks::Buffer blur;
	} uniformBuffers;

	struct {
		vks::Buffer sceneDesc;
		vks::Buffer firstPrimitives;
		vks::Buffer materials;
		vks::Buffer materialIndices;
		vks::Buffer reservoirs;
		vks::Buffer indirectReservoirs;
		vks::Buffer previousReservoirs;
		vks::Buffer previousIndirectReservoirs;
		vks::Buffer incidentRadianceGrid;
		vks::Buffer boundingVoxels;
		vks::Buffer voxelDebugStaging;  // Debug: staging buffer for reading voxel data
		vks::Buffer gmmStatisticsPack0;
		vks::Buffer gmmStatisticsPack1;
		vks::Buffer gmmStatisticsPack0Prev;
		vks::Buffer gmmStatisticsPack1Prev;
		vks::Buffer vpls;
		vks::Buffer screen;
	} storageBuffers;

	struct IncidentRadianceGridCell
	{
		float incidentRadianceSum[incidentRadianceMapSize * incidentRadianceMapSize];
		uint32_t incidentRadianceCount[incidentRadianceMapSize * incidentRadianceMapSize];
		float incidentRadiance[incidentRadianceMapSize * incidentRadianceMapSize];
		float cdf[incidentRadianceMapSize * incidentRadianceMapSize];
	};

	struct BoundingVoxel
	{
		uint32_t aabbMinX;
		uint32_t aabbMinY;
		uint32_t aabbMinZ;
		float totalIrradiance;
		uint32_t aabbMaxX;
		uint32_t aabbMaxY;
		uint32_t aabbMaxZ;
		uint32_t sampleCount;
	};

	std::vector<VulkanRaytracingSample::AccelerationStructure> bottomLevelASList;
	VulkanRaytracingSample::AccelerationStructure topLevelAS{};

	VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures{};
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enabledShaderAtomicFloatFeatures{};
	VkPhysicalDeviceVulkan11Features enabledVulkan11Features{};
	VkPhysicalDeviceVulkan12Features enabledVulkan12Features{};
	VkPhysicalDeviceVulkan13Features enabledVulkan13Features{};

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkFormat format;
		void destroy(VkDevice device)
		{
			vkDestroyImage(device, image, nullptr);
			vkDestroyImageView(device, view, nullptr);
			vkFreeMemory(device, mem, nullptr);
		}
	};
	struct FrameBuffer {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		VkRenderPass renderPass;
		void setSize(int32_t w, int32_t h)
		{
			this->width = w;
			this->height = h;
		}
		void destroy(VkDevice device)
		{
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
			vkDestroyRenderPass(device, renderPass, nullptr);
		}
	};

	struct {
		struct : public FrameBuffer {
			FrameBufferAttachment position, normal, albedo, specular, motion, depth;
		} gBuffer;
		struct : public FrameBuffer {
			FrameBufferAttachment color;
			FrameBufferAttachment history;
		} rayTracing;
		struct : public FrameBuffer {
			FrameBufferAttachment color;
		} blur;
		struct : public FrameBuffer {
			FrameBufferAttachment depth;
			FrameBufferAttachment history;
			FrameBufferAttachment color;
		} previous;
	} frameBuffers;

	// NRD input textures (for storing shader outputs)
	struct {
		FrameBufferAttachment radianceHitDist;   // R16G16B16A16_SFLOAT
		FrameBufferAttachment normalRoughness;   // R16G16B16A16_SFLOAT
		FrameBufferAttachment motionVector;      // R16G16_SFLOAT
		FrameBufferAttachment viewZ;             // R32_SFLOAT
	} nrdInputs;

	// One sampler for the frame buffer color attachments
	VkSampler colorSampler;

	VkQueryPool queryPool = VK_NULL_HANDLE;
	VkQueryPool computeQueryPool = VK_NULL_HANDLE;
	std::vector<uint64_t> timestamps;
	std::vector<uint64_t> computeTimestamps;
	double sumTimings[20] = { 0.0 };

	VulkanExample() : VulkanRaytracingSample()
	{
		title = "Path Guiding";
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 15.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif
		camera.position = { 2.15f, 0.3f, -8.75f };
		camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
		camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
		enableExtensions();
		enabledDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
	}

	~VulkanExample()
	{
		vkDestroySampler(device, colorSampler, nullptr);

		// Attachments
		frameBuffers.gBuffer.position.destroy(device);
		frameBuffers.gBuffer.normal.destroy(device);
		frameBuffers.gBuffer.albedo.destroy(device);
		frameBuffers.gBuffer.specular.destroy(device);
		frameBuffers.gBuffer.motion.destroy(device);
		frameBuffers.gBuffer.depth.destroy(device);
		frameBuffers.rayTracing.color.destroy(device);
		frameBuffers.rayTracing.history.destroy(device);
		frameBuffers.blur.color.destroy(device);
		frameBuffers.previous.depth.destroy(device);
		frameBuffers.previous.history.destroy(device);
		frameBuffers.previous.color.destroy(device);
		
		// NRD inputs
		nrdInputs.radianceHitDist.destroy(device);
		nrdInputs.normalRoughness.destroy(device);
		nrdInputs.motionVector.destroy(device);
		nrdInputs.viewZ.destroy(device);

		// Framebuffers
		frameBuffers.gBuffer.destroy(device);
		frameBuffers.rayTracing.destroy(device);
		frameBuffers.blur.destroy(device);
		frameBuffers.previous.destroy(device);
		
		// NRD wrapper cleanup
		nrdDenoiser.destroy();

		vkDestroySemaphore(device, semaphore, nullptr);

		vkDestroyPipeline(device, pipelines.gBuffer, nullptr);
		vkDestroyPipeline(device, pipelines.composition, nullptr);
		vkDestroyPipeline(device, pipelines.rayTracing, nullptr);
		vkDestroyPipeline(device, pipelines.blur, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayouts.gBuffer, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.rayTracing, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.blur, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.composition, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.gBuffer, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.rayTracing, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.blur, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.composition, nullptr);

		// Uniform buffers
		uniformBuffers.gBuffer.destroy();
		uniformBuffers.composition.destroy();
		uniformBuffers.blur.destroy();
		compute.uniformBuffer.destroy();

		storageBuffers.sceneDesc.destroy();
		storageBuffers.firstPrimitives.destroy();
		storageBuffers.materials.destroy();
		storageBuffers.materialIndices.destroy();
		storageBuffers.incidentRadianceGrid.destroy();
		storageBuffers.boundingVoxels.destroy();
		if (storageBuffers.voxelDebugStaging.buffer != VK_NULL_HANDLE)
			storageBuffers.voxelDebugStaging.destroy();

		textures.envMap.destroy();
		textures.envHDRCache.destroy();
		textures.whiteNoiseMap.destroy();
		textures.blueNoiseMap.destroy();

		for (auto &blas : bottomLevelASList)
			deleteAccelerationStructure(blas);
		deleteAccelerationStructure(topLevelAS);

		vkDestroyPipeline(device, compute.pipelines.reset, nullptr);
		vkDestroyPipeline(device, compute.pipelines.prepare, nullptr);
		vkDestroyPipeline(device, compute.pipelines.resetVXPG, nullptr);
		vkDestroyPipeline(device, compute.pipelines.prepareVXPG, nullptr);

		vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
		vkDestroySemaphore(device, compute.semaphore, nullptr);
		vkDestroyCommandPool(device, compute.commandPool, nullptr);

		vkDestroyQueryPool(device, queryPool, NULL);
		vkDestroyQueryPool(device, computeQueryPool, NULL);
	}

/*
	Create the bottom level acceleration structure contains the scene's actual geometry (vertices, triangles)
*/
	void createBottomLevelAccelerationStructure(VulkanRaytracingSample::AccelerationStructure &blas, 
		const vkglTF::Model &model, VkBuildAccelerationStructureFlagsKHR buildFlags, bool update = false)
	{
		if (update)
			deleteAccelerationStructure(blas);

		VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
		VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

		vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(model.vertices.buffer);
		indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(model.indices.buffer);

		std::vector<uint32_t> primitiveCounts;
		std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometries;
		VkDeviceOrHostAddressConstKHR indexBufferDeviceAddressOffset{};
		for (const auto node : model.linearNodes)
		{
			if (node->mesh)
				for (const auto primitive : node->mesh->primitives)
				{
					// Build
					VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
					if (primitive->material.transmissionFactor < 0.001f && (primitive->material.alphaMode == vkglTF::Material::ALPHAMODE_OPAQUE || (primitive->material.baseColorImage < 0 && primitive->material.baseColorFactor.w > 0.9f)))
						accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
					else
						accelerationStructureGeometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
					accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
					accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
					accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
					accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
					accelerationStructureGeometry.geometry.triangles.maxVertex = model.vertices.count;
					accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(vkglTF::Vertex);
					accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
					indexBufferDeviceAddressOffset.deviceAddress = indexBufferDeviceAddress.deviceAddress + primitive->firstIndex * sizeof(uint32_t);
					accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddressOffset;
					accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
					accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;
					accelerationStructureGeometries.emplace_back(accelerationStructureGeometry);
					primitiveCounts.emplace_back(primitive->indexCount / 3);
				}
		}
		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = buildFlags;
		accelerationStructureBuildGeometryInfo.geometryCount = accelerationStructureGeometries.size();
		accelerationStructureBuildGeometryInfo.pGeometries = accelerationStructureGeometries.data();

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			primitiveCounts.data(),
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructure(blas, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the bottom level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);
		
		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = buildFlags;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = blas.handle;
		accelerationBuildGeometryInfo.geometryCount = accelerationStructureGeometries.size();
		accelerationBuildGeometryInfo.pGeometries = accelerationStructureGeometries.data();
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;
		
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos{};
		for (auto i = 0; i < primitiveCounts.size(); ++i)
		{
			VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
			accelerationStructureBuildRangeInfo.primitiveCount = primitiveCounts[i];
			accelerationStructureBuildRangeInfo.primitiveOffset = 0;
			accelerationStructureBuildRangeInfo.firstVertex = 0;
			accelerationStructureBuildRangeInfo.transformOffset = 0;
			accelerationStructureBuildRangeInfos.emplace_back(accelerationStructureBuildRangeInfo);
		}
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos;
		for (auto &info : accelerationStructureBuildRangeInfos)
			accelerationBuildStructureRangeInfos.emplace_back(&info);

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);
		
		deleteScratchBuffer(scratchBuffer);
	}
	
	/*
		The top level acceleration structure contains the scene's object instances
	*/
	void createTopLevelAccelerationStructure(VkBuildAccelerationStructureFlagsKHR buildFlags, bool update = false)
	{
		VkTransformMatrixKHR transformMatrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f };

		std::vector<VkAccelerationStructureInstanceKHR> instances(bottomLevelASList.size());
		for (auto i = 0; i < instances.size(); ++i)
		{
			instances[i].transform = transformMatrix;
			instances[i].instanceCustomIndex = i;
			instances[i].mask = 0xFF;
			instances[i].instanceShaderBindingTableRecordOffset = i;
			instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;//VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR
			instances[i].accelerationStructureReference = bottomLevelASList[i].deviceAddress;
		}

		// Buffer for instance data
		vks::Buffer instancesBuffer;
		createBuffer(instancesBuffer, instances.size() * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instances.data());

		VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
		instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer.buffer);
		
		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
		accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;
		
		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = buildFlags;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		uint32_t primitive_count = instances.size();

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&primitive_count,
			&accelerationStructureBuildSizesInfo);

		if (!update)
			createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the top level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = buildFlags;
		accelerationBuildGeometryInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : 
			VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.srcAccelerationStructure = update ? topLevelAS.handle : VK_NULL_HANDLE;
		accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = instances.size();
		accelerationStructureBuildRangeInfo.primitiveOffset = 0;
		accelerationStructureBuildRangeInfo.firstVertex = 0;
		accelerationStructureBuildRangeInfo.transformOffset = 0;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		deleteScratchBuffer(scratchBuffer);
		instancesBuffer.destroy();
	}

	void getEnabledFeatures()
	{
		// Enable anisotropic filtering if supported
		if (deviceFeatures.samplerAnisotropy)
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		if (deviceFeatures.shaderInt64)
			enabledFeatures.shaderInt64 = VK_TRUE;
		if (deviceFeatures.fragmentStoresAndAtomics)
			enabledFeatures.fragmentStoresAndAtomics = VK_TRUE;
		if (deviceFeatures.geometryShader)
			enabledFeatures.geometryShader = VK_TRUE;

		enabledVulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		enabledVulkan13Features.maintenance4 = VK_TRUE;
		enabledVulkan13Features.pNext = nullptr;
		
		enabledVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		enabledVulkan12Features.bufferDeviceAddress = VK_TRUE;
		enabledVulkan12Features.storageBuffer8BitAccess = VK_TRUE;
		enabledVulkan12Features.shaderInt8 = VK_TRUE;
		enabledVulkan12Features.shaderFloat16 = VK_TRUE;
		enabledVulkan12Features.scalarBlockLayout = VK_TRUE;
		enabledVulkan12Features.runtimeDescriptorArray = VK_TRUE;
		enabledVulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		enabledVulkan12Features.descriptorIndexing = VK_TRUE;
		enabledVulkan12Features.pNext = &enabledVulkan13Features;

		enabledVulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		enabledVulkan11Features.storageBuffer16BitAccess = VK_TRUE;
		enabledVulkan11Features.pNext = &enabledVulkan12Features;

		enabledShaderAtomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
		enabledShaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
		enabledShaderAtomicFloatFeatures.pNext = &enabledVulkan11Features;

		enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
		enabledAccelerationStructureFeatures.pNext = &enabledShaderAtomicFloatFeatures;

		enabledRayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
		enabledRayQueryFeatures.rayQuery = VK_TRUE;
		enabledRayQueryFeatures.pNext = &enabledAccelerationStructureFeatures;

		deviceCreatepNextChain = &enabledRayQueryFeatures;

		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		timeStampPeriod = properties.limits.timestampPeriod / 1e6f;
	}

	// Create a frame buffer attachment
	void createAttachment(
		VkFormat format,
		VkImageUsageFlagBits usage,
		FrameBufferAttachment *attachment,
		uint32_t width,
		uint32_t height)
	{
		VkImageAspectFlags aspectMask = 0;

		attachment->format = format;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
			aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
				aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}

		assert(aspectMask > 0);

		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = format;
		image.extent.width = width;
		image.extent.height = height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
		vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageView.format = format;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
	}

	void prepareOffscreenFramebuffers()
	{
		// Attachments
#if defined(__ANDROID__)
		const uint32_t ssaoWidth = width / 2;
		const uint32_t ssaoHeight = height / 2;
#else
		const uint32_t ssaoWidth = width;
		const uint32_t ssaoHeight = height;
#endif

		frameBuffers.gBuffer.setSize(width, height);
		frameBuffers.rayTracing.setSize(ssaoWidth, ssaoHeight);
		frameBuffers.blur.setSize(width, height);
		frameBuffers.previous.setSize(ssaoWidth, ssaoHeight);

		// Find a suitable depth format
		VkFormat attDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);

		// G-Buffer
		createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.gBuffer.position, width, height);	// Position + Depth
		createAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.gBuffer.normal, width, height);			// Normals
		createAttachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.gBuffer.albedo, width, height);			// Albedo (color)
		createAttachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.gBuffer.specular, width, height);
		createAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.gBuffer.motion, width, height);
		createAttachment(attDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &frameBuffers.gBuffer.depth, width, height);			// Depth

		// SSAO
		createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.rayTracing.color, ssaoWidth, ssaoHeight);				// Color
		createAttachment(VK_FORMAT_R8_UINT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.rayTracing.history, width, height);

		// SSAO blur (需要STORAGE_BIT因为NRD会写入它)
		createAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, (VkImageUsageFlagBits)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT), &frameBuffers.blur.color, width, height);

		createAttachment(VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.previous.depth, ssaoWidth, ssaoHeight);
		createAttachment(VK_FORMAT_R8_UINT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.previous.history, ssaoWidth, ssaoHeight);

		createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &frameBuffers.previous.color, ssaoWidth, ssaoHeight);

		// NRD input textures (需要STORAGE_BIT用于compute shader)
		createAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, (VkImageUsageFlagBits)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT), &nrdInputs.radianceHitDist, ssaoWidth, ssaoHeight);
		createAttachment(VK_FORMAT_R16G16B16A16_SFLOAT, (VkImageUsageFlagBits)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT), &nrdInputs.normalRoughness, ssaoWidth, ssaoHeight);
		createAttachment(VK_FORMAT_R16G16_SFLOAT, (VkImageUsageFlagBits)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT), &nrdInputs.motionVector, ssaoWidth, ssaoHeight);
		createAttachment(VK_FORMAT_R32_SFLOAT, (VkImageUsageFlagBits)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT), &nrdInputs.viewZ, ssaoWidth, ssaoHeight);
		
		// 注意：NRD的history buffer layout已经在NRDWrapper::createHistoryTexture()中初始化到GENERAL
		// 其他textures的layout会在第一次render pass中自动转换

		// Render passes

		// G-Buffer creation
		{
			std::array<VkAttachmentDescription, 6> attachmentDescs = {};

			// Init attachment properties
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++)
			{
				attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
				attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
				attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
				attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
				attachmentDescs[i].finalLayout = (i == attachmentDescs.size() - 1) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}

			// Formats
			attachmentDescs[0].format = frameBuffers.gBuffer.position.format;
			attachmentDescs[1].format = frameBuffers.gBuffer.normal.format;
			attachmentDescs[2].format = frameBuffers.gBuffer.albedo.format;
			attachmentDescs[3].format = frameBuffers.gBuffer.specular.format;
			attachmentDescs[4].format = frameBuffers.gBuffer.motion.format;
			attachmentDescs[5].format = frameBuffers.gBuffer.depth.format;

			std::vector<VkAttachmentReference> colorReferences;
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size() - 1); ++i)
				colorReferences.push_back({ i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

			VkAttachmentReference depthReference = {};
			depthReference.attachment = attachmentDescs.size() - 1;
			depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.pColorAttachments = colorReferences.data();
			subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
			subpass.pDepthStencilAttachment = &depthReference;

			// Use subpass dependencies for attachment layout transitions
			std::array<VkSubpassDependency, 2> dependencies;

			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			dependencies[1].srcSubpass = 0;
			dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			VkRenderPassCreateInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.pAttachments = attachmentDescs.data();
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();
			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &frameBuffers.gBuffer.renderPass));

			std::array<VkImageView, static_cast<uint32_t>(attachmentDescs.size())> attachments;
			attachments[0] = frameBuffers.gBuffer.position.view;
			attachments[1] = frameBuffers.gBuffer.normal.view;
			attachments[2] = frameBuffers.gBuffer.albedo.view;
			attachments[3] = frameBuffers.gBuffer.specular.view;
			attachments[4] = frameBuffers.gBuffer.motion.view;
			attachments[5] = frameBuffers.gBuffer.depth.view;

			VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.gBuffer.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = frameBuffers.gBuffer.width;
			fbufCreateInfo.height = frameBuffers.gBuffer.height;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuffers.gBuffer.frameBuffer));
		}

		// SSAO (Ray Tracing with NRD outputs)
		{
			std::array<VkAttachmentDescription, 6> attachmentDescs = {};

			// Init attachment properties
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++)
			{
				attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
				attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
				attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
				attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}

			// Attachment 0: Color
			attachmentDescs[0].format = frameBuffers.rayTracing.color.format;
			// Attachment 1: History
			attachmentDescs[1].format = frameBuffers.rayTracing.history.format;
			// Attachment 2: Radiance + Hit Distance (NRD)
			attachmentDescs[2].format = nrdInputs.radianceHitDist.format;
			// Attachment 3: Normal + Roughness (NRD)
			attachmentDescs[3].format = nrdInputs.normalRoughness.format;
			// Attachment 4: Motion Vector (NRD)
			attachmentDescs[4].format = nrdInputs.motionVector.format;
			// Attachment 5: ViewZ (NRD)
			attachmentDescs[5].format = nrdInputs.viewZ.format;

			std::vector<VkAttachmentReference> colorReferences;
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); ++i)
				colorReferences.push_back({ i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.pColorAttachments = colorReferences.data();
			subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());

			std::array<VkSubpassDependency, 2> dependencies;

			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			dependencies[1].srcSubpass = 0;
			dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			VkRenderPassCreateInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.pAttachments = attachmentDescs.data();
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();
			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &frameBuffers.rayTracing.renderPass));

			std::array<VkImageView, static_cast<uint32_t>(attachmentDescs.size())> attachments;
			attachments[0] = frameBuffers.rayTracing.color.view;
			attachments[1] = frameBuffers.rayTracing.history.view;
			attachments[2] = nrdInputs.radianceHitDist.view;
			attachments[3] = nrdInputs.normalRoughness.view;
			attachments[4] = nrdInputs.motionVector.view;
			attachments[5] = nrdInputs.viewZ.view;

			VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.rayTracing.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = frameBuffers.rayTracing.width;
			fbufCreateInfo.height = frameBuffers.rayTracing.height;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuffers.rayTracing.frameBuffer));
		}

		// SSAO Blur
		{
			std::array<VkAttachmentDescription, 4> attachmentDescs = {};

			// Init attachment properties
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++)
			{
				attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
				attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
				attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
				attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}

			// Formats
			attachmentDescs[0].format = frameBuffers.blur.color.format;
			attachmentDescs[1].format = frameBuffers.previous.depth.format;
			attachmentDescs[2].format = frameBuffers.previous.history.format;
			attachmentDescs[3].format = frameBuffers.previous.color.format;

			std::vector<VkAttachmentReference> colorReferences;
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); ++i)
				colorReferences.push_back({ i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.pColorAttachments = colorReferences.data();
			subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());

			std::array<VkSubpassDependency, 2> dependencies;

			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			dependencies[1].srcSubpass = 0;
			dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			VkRenderPassCreateInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.pAttachments = attachmentDescs.data();
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();
			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &frameBuffers.blur.renderPass));

			std::array<VkImageView, static_cast<uint32_t>(attachmentDescs.size())> attachments;
			attachments[0] = frameBuffers.blur.color.view;
			attachments[1] = frameBuffers.previous.depth.view;
			attachments[2] = frameBuffers.previous.history.view;
			attachments[3] = frameBuffers.previous.color.view;

			VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.blur.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = frameBuffers.blur.width;
			fbufCreateInfo.height = frameBuffers.blur.height;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuffers.blur.frameBuffer));
		}

		// Shared sampler used for all color attachments
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_NEAREST;
		sampler.minFilter = VK_FILTER_NEAREST;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &colorSampler));
	}

	void loadAssets()
	{
		vkglTF::memoryPropertyFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices;

		models.models.resize(1);
		
		if (settings.scene == 0)
			models.models.front().loadFromFile("D:/PG/VulkanReSTIR/assets/models/Sponza/glTF/Sponza.gltf", vulkanDevice, queue, glTFLoadingFlags);// , glm::scale(glm::mat4(1.0), glm::vec3(0.0254f / 0.008f)));

		if (settings.scene == 1)
			models.models.front().loadFromFile("D:/packman-repo/chk/rtxdi-media/p4sw-30874147/bistro/bistro.gltf", vulkanDevice, queue, glTFLoadingFlags);
		
		if (settings.scene == 2)
			models.models.front().loadFromFile("D:/PG/VulkanPT/assets/models/San_Miguel/sanmiguel.gltf", vulkanDevice, queue, glTFLoadingFlags);
		
		//为什么不会被光线击中？
		//models.models.back().loadFromFile(getAssetPath() + "models/sphere.gltf", vulkanDevice, queue, glTFLoadingFlags, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-5.279618f, 2.332777f, 11.332270f)), glm::vec3(0.2f)));

		if (settings.scene == 0)
			textures.envMap.loadFromFile("D:/PG/VulkanReSTIR/assets/textures/uffizi-large.ktx2", VK_FORMAT_R16G16B16A16_SFLOAT, vulkanDevice, queue);
		else
			textures.envMap.loadFromFile(getAssetPath() + "textures/pisa.ktx2", VK_FORMAT_R16G16B16A16_SFLOAT, vulkanDevice, queue);
		textures.envHDRCache.fromBuffer(textures.envMap.hdrCache.data(), textures.envMap.hdrCache.size() * sizeof(float), VK_FORMAT_R32G32B32A32_SFLOAT, textures.envMap.width, textures.envMap.height, vulkanDevice, queue, VK_FILTER_NEAREST);

		struct Material
		{
			glm::vec4 baseColor;
			float metallic;
			float roughness;
			int baseColorImage;
			int metallicRoughnessImage;
			int normalImage;
			int emissiveImage;
			int32_t alphaMode;
			float transmission;
			bool metallicRoughness;
			glm::vec3 specularFactor;
		};

		std::vector<Material> materials;
		std::vector<int> materialIndices;
		std::vector<int> firstPrimitives;

		for (const auto &model : models.models)
		{
			for (const auto &material : model.materials)
			{
				materials.emplace_back();
				materials.back().baseColor = material.baseColorFactor;
				materials.back().metallic = material.metallicFactor;
				materials.back().roughness = material.roughnessFactor;
				materials.back().baseColorImage = material.baseColorImage;
				materials.back().metallicRoughnessImage = material.metallicRoughnessImage;
				materials.back().normalImage = material.normalImage;
				materials.back().emissiveImage = material.emissiveImage;
				materials.back().alphaMode = material.alphaMode;
				materials.back().transmission = material.transmissionFactor;
				materials.back().metallicRoughness = material.metallicRoughness;
				materials.back().specularFactor = material.specularFactor;
			}

			for (const auto &node : model.linearNodes)
				if (node->mesh)
				{
					for (auto primitive : node->mesh->primitives)
					{
						for (auto i = 0; i < primitive->indexCount / 3; ++i)
							//这里加materialOffset没用，因为第二个物体访问不到后面的materialIndices，而且要用materialIndex是否为0判断是否有材质
							materialIndices.emplace_back(primitive->materialIndex);
						firstPrimitives.emplace_back(primitive->firstIndex / 3);
					}
				}
		}

		createBuffer(storageBuffers.firstPrimitives, firstPrimitives.size() * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, firstPrimitives.data());
		createBuffer(storageBuffers.materials, materials.size() * sizeof(Material), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, materials.data());
		createBuffer(storageBuffers.materialIndices, materialIndices.size() * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, materialIndices.data());

		struct ModelInstance
		{
			uint32_t modelIndex{ 0 };
			uint32_t imageOffset{ 0 };
			VkDeviceAddress vertexAddress;
			VkDeviceAddress indexAddress;
			VkDeviceAddress materialAddress;
			VkDeviceAddress materialIndexAddress;
			VkDeviceAddress firstPrimitiveAddress;
		};
		std::vector<ModelInstance> modelInstances;

		auto imageOffset = 0;
		auto materialOffset = 0;
		auto materialIndexOffset = 0;
		auto firstPrimitiveOffset = 0;
		for (auto i = 0; i < models.models.size(); ++i)
		{
			ModelInstance instance;
			instance.modelIndex = i;
			instance.imageOffset = imageOffset;
			instance.vertexAddress = getBufferDeviceAddress(models.models[i].vertices.buffer);
			instance.indexAddress = getBufferDeviceAddress(models.models[i].indices.buffer);
			instance.materialAddress = getBufferDeviceAddress(storageBuffers.materials.buffer) + materialOffset * sizeof(Material);
			instance.materialIndexAddress = getBufferDeviceAddress(storageBuffers.materialIndices.buffer) + materialIndexOffset * sizeof(int);
			instance.firstPrimitiveAddress = getBufferDeviceAddress(storageBuffers.firstPrimitives.buffer) + firstPrimitiveOffset * sizeof(int);
			modelInstances.emplace_back(instance);
			imageOffset += models.models[i].images.size();
			materialOffset += models.models[i].materials.size();
			for (const auto &node : models.models[i].linearNodes)
				if (node->mesh)
					for (auto primitive : node->mesh->primitives)
					{
						materialIndexOffset += primitive->indexCount / 3;
						++firstPrimitiveOffset;
					}
		}

		createBuffer(storageBuffers.sceneDesc, modelInstances.size() * sizeof(ModelInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, modelInstances.data());

		auto imageWidth = 0;
		auto imageHeight = 0;
		auto imageChannels = 0;
		auto blueNoise = stbi_load((getAssetPath() + "textures/LDR_RG01_0.png").c_str(), &imageWidth, &imageHeight, &imageChannels, 0);

		// Random noise
		std::vector<byte> noiseBuffer(width * height * 2);
		for (auto &noise : noiseBuffer)
			noise = (float)rand() / RAND_MAX * 255;
		textures.whiteNoiseMap.fromBuffer(noiseBuffer.data(), noiseBuffer.size(), VK_FORMAT_R8G8_UNORM, width, height, vulkanDevice, queue, VK_FILTER_NEAREST);
		for (auto j = 0; j < height; ++j)
			for (auto i = 0; i < width; ++i)
			{
				noiseBuffer[j * width * 2 + i * 2] = blueNoise[j * imageHeight * imageChannels + i * imageChannels];
				noiseBuffer[j * width * 2 + i * 2 + 1] = 255 - blueNoise[j * imageHeight * imageChannels + i * imageChannels + 1];
			}
		stbi_image_free(blueNoise);
		textures.blueNoiseMap.fromBuffer(noiseBuffer.data(), noiseBuffer.size(), VK_FORMAT_R8G8_UNORM, width, height, vulkanDevice, queue, VK_FILTER_NEAREST);

		struct PackedReservoir
		{
			uint32_t sampleSeed;
			uint16_t W;
			uint32_t M;	//样本数量
		};

		struct PackedIndirectReservoir
		{
			glm::vec3 position;
			glm::u16vec3 normal;
			glm::u16vec3 radiance;
			uint16_t pdf;
			uint16_t W;	//RIS
			uint32_t M;	//样本数量
		};

		createBuffer(storageBuffers.reservoirs, width * height * sizeof(PackedReservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.previousReservoirs, width * height * sizeof(PackedReservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.indirectReservoirs, width * height * sizeof(PackedIndirectReservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.previousIndirectReservoirs, width * height * sizeof(PackedIndirectReservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		
		glm::vec3 end;
		if (settings.scene == 0)
		{
			pushConstants.gridBegin = glm::vec3(-15.895318f, -1.125410f, -9.818801f);
			end = glm::vec3(14.613639f, 11.694201f, 8.935804f);
		}
		if (settings.scene == 1)
		{
			pushConstants.gridBegin = glm::vec3(-44.844467f, -0.5f, -15.579536f);
			end = glm::vec3(25.099527f, 24.225739f, 29.100180f);
		}
		if (settings.scene == 2)
		{
			pushConstants.gridBegin = glm::vec3(-1.150779f, -0.301780f, -11.600891f);
			end = glm::vec3(32.703480f, 15.362855f, 14.100478f);
		}

		auto gridSize = glm::ceil((end - pushConstants.gridBegin) / pushConstants.cellSize);
		pushConstants.gridDim = glm::ivec3(gridSize.x, gridSize.y, gridSize.z);
		printf("%d %d %d\n", pushConstants.gridDim.x, pushConstants.gridDim.y, pushConstants.gridDim.z);

		createBuffer(storageBuffers.incidentRadianceGrid, pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z * sizeof(IncidentRadianceGridCell), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.boundingVoxels, pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z * sizeof(BoundingVoxel), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);  // Add TRANSFER_SRC for debug reading
	
		createBuffer(storageBuffers.gmmStatisticsPack0, width * height * pushConstants.lobeCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.gmmStatisticsPack1, width * height * pushConstants.lobeCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.gmmStatisticsPack0Prev, width * height * pushConstants.lobeCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.gmmStatisticsPack1Prev, width * height * pushConstants.lobeCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(storageBuffers.vpls, width * height * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	
		createBuffer(storageBuffers.screen, width * height * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}
	
	// Setup a query pool for storing pipeline statistics
	void setupQueryPool()
	{
		VkQueryPoolCreateInfo queryPoolInfo = {};
		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolInfo.queryCount = timestamps.size();
		VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, NULL, &queryPool));

		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdResetQueryPool(commandBuffer, queryPool, 0, queryPoolInfo.queryCount);

		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolInfo.queryCount = computeTimestamps.size();
		VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, NULL, &computeQueryPool));

		vkCmdResetQueryPool(commandBuffer, computeQueryPool, 0, queryPoolInfo.queryCount);
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);
	}

	// Retrieves the results of the pipeline statistics query submitted to the command buffer
	void getQueryResults()
	{
		// We use vkGetQueryResults to copy the results into a host visible buffer
		vkGetQueryPoolResults(
			device,
			queryPool,
			0,
			timestamps.size(),
			sizeof(uint64_t) * timestamps.size(),
			timestamps.data(),
			sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT);
			
		vkGetQueryPoolResults(
			device,
			computeQueryPool,
			0,
			computeTimestamps.size(),
			sizeof(uint64_t) * computeTimestamps.size(),
			computeTimestamps.data(),
			sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT);
	}

	void buildCommandBuffers() override
	{
		static int buildCount = 0;
		std::cout << "[DEBUG] buildCommandBuffers called, count=" << ++buildCount 
		          << ", nrdFrameIndex=" << nrdFrameIndex << std::endl;
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			
			// 注意：NRD相关images的layout初始化已经在prepareOffscreenFramebuffers中完成
			// 这里不需要重复初始化

			/*
				Offscreen SSAO generation
			*/
			{
				// Clear values for all attachments written in the fragment shader
				std::vector<VkClearValue> clearValues(6);
				clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[3].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[4].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[5].depthStencil = { 1.0f, 0 };

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = frameBuffers.gBuffer.renderPass;
				renderPassBeginInfo.framebuffer = frameBuffers.gBuffer.frameBuffer;
				renderPassBeginInfo.renderArea.extent.width = frameBuffers.gBuffer.width;
				renderPassBeginInfo.renderArea.extent.height = frameBuffers.gBuffer.height;
				renderPassBeginInfo.clearValueCount = 6;
				renderPassBeginInfo.pClearValues = clearValues.data();

				/*
					First pass: Fill G-Buffer components (positions+depth, normals, albedo) using MRT
				*/

				vkCmdResetQueryPool(drawCmdBuffers[i], queryPool, i * numTimestamps, numTimestamps);
				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, i * numTimestamps + 8);
				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, i * numTimestamps);
				
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				VkViewport viewport = vks::initializers::viewport((float)frameBuffers.gBuffer.width, (float)frameBuffers.gBuffer.height, 0.0f, 1.0f);
				viewport.y = viewport.height;
				viewport.height = -viewport.height;
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				VkRect2D scissor = vks::initializers::rect2D(frameBuffers.gBuffer.width, frameBuffers.gBuffer.height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.gBuffer);

				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.gBuffer, 0, 1, &descriptorSets.model, 0, nullptr);
				for (auto j = 0; j < models.models.size(); ++j)
				{
					pushConstants.modelIndex = j;
					vkCmdPushConstants(drawCmdBuffers[i], pipelineLayouts.gBuffer, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);
					models.models[j].draw(drawCmdBuffers[i]);
				}

				vkCmdEndRenderPass(drawCmdBuffers[i]);

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i * numTimestamps + 1);

				/*
					Second pass: Ray Tracing with NRD outputs
				*/

				// Prepare clear values for Ray Tracing pass (6 attachments)
				clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };  // color
				clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };  // history
				clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };  // radiance+hitDist (NRD)
				clearValues[3].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };  // normal+roughness (NRD)
				clearValues[4].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };  // motionVector (NRD)
				clearValues[5].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };  // viewZ (NRD)

				renderPassBeginInfo.framebuffer = frameBuffers.rayTracing.frameBuffer;
				renderPassBeginInfo.renderPass = frameBuffers.rayTracing.renderPass;
				renderPassBeginInfo.renderArea.extent.width = frameBuffers.rayTracing.width;
				renderPassBeginInfo.renderArea.extent.height = frameBuffers.rayTracing.height;
				renderPassBeginInfo.clearValueCount = 6;  // Changed from 3 to 6 for NRD outputs
				renderPassBeginInfo.pClearValues = clearValues.data();

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, i * numTimestamps + 2);

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				viewport = vks::initializers::viewport((float)frameBuffers.rayTracing.width, (float)frameBuffers.rayTracing.height, 0.0f, 1.0f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				scissor = vks::initializers::rect2D(frameBuffers.rayTracing.width, frameBuffers.rayTracing.height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.rayTracing, 0, 1, &descriptorSets.rayTracing, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.rayTracing);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

				vkCmdEndRenderPass(drawCmdBuffers[i]);

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i * numTimestamps + 3);

				/*
					Third pass: SSAO blur
				*/
				
				for (auto i = 0; i < 4; ++i)
					clearValues[i].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
				clearValues[4].depthStencil = { 1.0f, 0 };

				renderPassBeginInfo.framebuffer = frameBuffers.blur.frameBuffer;
				renderPassBeginInfo.renderPass = frameBuffers.blur.renderPass;
				renderPassBeginInfo.renderArea.extent.width = frameBuffers.blur.width;
				renderPassBeginInfo.renderArea.extent.height = frameBuffers.blur.height;
				renderPassBeginInfo.clearValueCount = 5;

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, i * numTimestamps + 4);

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				viewport = vks::initializers::viewport((float)frameBuffers.blur.width, (float)frameBuffers.blur.height, 0.0f, 1.0f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				scissor = vks::initializers::rect2D(frameBuffers.blur.width, frameBuffers.blur.height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blur, 0, 1, &descriptorSets.blur, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blur);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

				vkCmdEndRenderPass(drawCmdBuffers[i]);

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i * numTimestamps + 5);
			}

			/*
				NRD Denoising pass (if enabled)
			*/
			if (useNRD && nrdDenoiser.isInitialized())
			{
				// 设置NRD的输入和输出纹理
				nrdDenoiser.setInputTextures(
					nrdInputs.radianceHitDist.view,
					nrdInputs.normalRoughness.view,
					nrdInputs.viewZ.view,
					nrdInputs.motionVector.view
				);
				
				// 让NRD直接输出到blur的framebuffer，这样composition就能看到NRD的结果
				nrdDenoiser.setOutputTexture(frameBuffers.blur.color.view);
				
				// 转换image layouts到GENERAL（compute shader需要）
				VkImageMemoryBarrier barriers[5] = {};
				
				// NRD inputs: SHADER_READ_ONLY_OPTIMAL -> GENERAL
				// Ray Tracing pass之后，这些attachments被转换到SHADER_READ_ONLY_OPTIMAL
				for (int j = 0; j < 4; j++)
				{
					barriers[j].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					barriers[j].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
					barriers[j].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
					barriers[j].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					barriers[j].newLayout = VK_IMAGE_LAYOUT_GENERAL;
					barriers[j].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barriers[j].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barriers[j].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					barriers[j].subresourceRange.baseMipLevel = 0;
					barriers[j].subresourceRange.levelCount = 1;
					barriers[j].subresourceRange.baseArrayLayer = 0;
					barriers[j].subresourceRange.layerCount = 1;
				}
				barriers[0].image = nrdInputs.radianceHitDist.image;
				barriers[1].image = nrdInputs.normalRoughness.image;
				barriers[2].image = nrdInputs.viewZ.image;
				barriers[3].image = nrdInputs.motionVector.image;
				
				// blur output: SHADER_READ_ONLY_OPTIMAL -> GENERAL
				barriers[4].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barriers[4].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
				barriers[4].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				barriers[4].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barriers[4].newLayout = VK_IMAGE_LAYOUT_GENERAL;
				barriers[4].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriers[4].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriers[4].image = frameBuffers.blur.color.image;
				barriers[4].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				barriers[4].subresourceRange.baseMipLevel = 0;
				barriers[4].subresourceRange.levelCount = 1;
				barriers[4].subresourceRange.baseArrayLayer = 0;
				barriers[4].subresourceRange.layerCount = 1;
				
				vkCmdPipelineBarrier(
					drawCmdBuffers[i],
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0,
					0, nullptr,
					0, nullptr,
					5, barriers
				);
				
				// Get current and previous matrices
				glm::mat4 currentView = camera.matrices.view;
				glm::mat4 currentProj = camera.matrices.perspective;
				
				// Store previous matrices (static to persist between frames)
				static glm::mat4 prevView = currentView;
				static glm::mat4 prevProj = currentProj;
				
				// 只在第一个command buffer时输出（避免重复）
				if (i == 0 && (nrdFrameIndex < 10 || nrdFrameIndex % 60 == 0))
				{
					std::cout << "[NRD] Continuous frame " << nrdFrameIndex << std::endl;
				}
				
				// Call NRD denoise
				nrdDenoiser.denoise(
					drawCmdBuffers[i],
					nrdFrameIndex,  // 使用成员变量nrdFrameIndex
					currentView,
					currentProj,
					prevView,
					prevProj
				);
				
				// Only update on first command buffer
				if (i == 0)
				{
					prevView = currentView;
					prevProj = currentProj;
				}
				
				// 转换image layouts回到原来的状态
				VkImageMemoryBarrier barriersBack[5] = {};
				
				// NRD inputs: GENERAL -> SHADER_READ_ONLY_OPTIMAL
				for (int j = 0; j < 4; j++)
				{
					barriersBack[j].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					barriersBack[j].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
					barriersBack[j].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
					barriersBack[j].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
					barriersBack[j].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					barriersBack[j].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barriersBack[j].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barriersBack[j].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					barriersBack[j].subresourceRange.baseMipLevel = 0;
					barriersBack[j].subresourceRange.levelCount = 1;
					barriersBack[j].subresourceRange.baseArrayLayer = 0;
					barriersBack[j].subresourceRange.layerCount = 1;
				}
				barriersBack[0].image = nrdInputs.radianceHitDist.image;
				barriersBack[1].image = nrdInputs.normalRoughness.image;
				barriersBack[2].image = nrdInputs.viewZ.image;
				barriersBack[3].image = nrdInputs.motionVector.image;
				
				// blur output: GENERAL -> SHADER_READ_ONLY_OPTIMAL
				barriersBack[4].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barriersBack[4].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				barriersBack[4].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				barriersBack[4].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
				barriersBack[4].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barriersBack[4].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriersBack[4].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriersBack[4].image = frameBuffers.blur.color.image;
				barriersBack[4].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				barriersBack[4].subresourceRange.baseMipLevel = 0;
				barriersBack[4].subresourceRange.levelCount = 1;
				barriersBack[4].subresourceRange.baseArrayLayer = 0;
				barriersBack[4].subresourceRange.layerCount = 1;
				
				vkCmdPipelineBarrier(
					drawCmdBuffers[i],
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					0,
					0, nullptr,
					0, nullptr,
					5, barriersBack
				);
			}

			/*
				Note: Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
			*/

			/*
				Final render pass: Scene rendering with applied radial blur
			*/
			{
				std::vector<VkClearValue> clearValues(2);
				clearValues[0].color = defaultClearColor;
				clearValues[1].depthStencil = { 1.0f, 0 };

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = renderPass;
				renderPassBeginInfo.framebuffer = VulkanExampleBase::frameBuffers[i];
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = 2;
				renderPassBeginInfo.pClearValues = clearValues.data();

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, i * numTimestamps + 6);

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.composition, 0, 1, &descriptorSets.composition, 0, nullptr);

				// Final composition pass
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.composition);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

				drawUI(drawCmdBuffers[i]);

				vkCmdEndRenderPass(drawCmdBuffers[i]);

				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i * numTimestamps + 7);
				vkCmdWriteTimestamp(drawCmdBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i * numTimestamps + 9);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void buildComputeCommandBuffers() override
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			auto &commandBuffer = compute.commandBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
			vkCmdPushConstants(commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);
			vkCmdResetQueryPool(commandBuffer, computeQueryPool, i * numComputeTimestamps, numComputeTimestamps);

			auto dispatchCompute = [this, &commandBuffer, &i](const VkPipeline &pipeline, glm::uvec3 groupCount, int timeStampOffset = -1)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
				if (timeStampOffset >= 0)
					vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, computeQueryPool, i * numComputeTimestamps + timeStampOffset);
				vkCmdDispatch(commandBuffer, groupCount.x, groupCount.y, groupCount.z);
				if (timeStampOffset >= 0)
					vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, computeQueryPool, i * numComputeTimestamps + timeStampOffset + 1);
			};

			auto addMemoryBarrier = [&commandBuffer](VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
			{
				VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
				memoryBarrier.srcAccessMask = srcAccessMask;
				memoryBarrier.dstAccessMask = dstAccessMask;
				vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
			};

			if (specializationData.vxpg == 1)
			{
				// VXPG: Reset voxel data at the start of each frame (per-frame learning)
				dispatchCompute(compute.pipelines.resetVXPG, glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 63u) / 64, 1, 1), -1);
				
				// Memory barrier: ensure reset completes before prepare
				addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
				                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
				
				// VXPG prepare
				dispatchCompute(compute.pipelines.prepareVXPG, glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 63u) / 64, 1, 1), 0);
			}
			else if (specializationData.sspg == 0)
			{
				if (specializationData.cdf == 0)
					dispatchCompute(compute.pipelines.prepare, glm::uvec3(pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z, 1, 1), 0);
				else
					dispatchCompute(compute.pipelines.prepare, glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 7u) / 8, 1, 1), 0);
			}
			else
			{
				dispatchCompute(compute.pipelines.prepare, glm::uvec3((width + 15u) / 16, (height + 15u) / 16, 1), 0);
			}
			
			vkEndCommandBuffer(commandBuffer);
		}
	}

	void setupDescriptorPool()
	{
		auto numImages = 0;
		for (const auto &model : models.models)
			numImages += model.images.size();
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 101),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numImages + 71),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 10),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 80)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, descriptorSets.count);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupLayoutsAndDescriptors()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo;
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo();
		VkDescriptorSetAllocateInfo descriptorAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, nullptr, 1);
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		std::vector<VkDescriptorImageInfo> imageDescriptors;

		auto numImages = 0;
		for (const auto &model : models.models)
			numImages += model.images.size();
		// G-Buffer creation (offscreen scene rendering)
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),	// VS + FS Parameter UBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2, numImages)
		};

		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.gBuffer));

		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.gBuffer;
		//// Push constants in the fragment shader
		VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants) };
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.gBuffer));
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.gBuffer;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.model));
		for (const auto &model : models.models)
			for (const auto &image : model.images)
				imageDescriptors.push_back(image.descriptor);
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.gBuffer.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &storageBuffers.sceneDesc.descriptor),
		};
		if (numImages > 0)
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageDescriptors[0], numImages));
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// Ray Tracing
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Position+Depth
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),						// FS Normals
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),						// FS SSAO Noise
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),						// FS Normals
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 5),						// FS SSAO Noise
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 6),								// FS SSAO Kernel UBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 7),	// FS Params UBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 8),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 9),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 10),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 11),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 12),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 13, numImages),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 14),
			//vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 15),
			//vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 16),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 17),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 18),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 19),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 20),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 21),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 22),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 23),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 24),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 25),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 26),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_FRAGMENT_BIT, 27),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 28),
		};
		
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.rayTracing));
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.rayTracing;
		pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
		pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.rayTracing));
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.rayTracing;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.rayTracing));
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.motion.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.previous.history.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.previous.depth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.previous.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.specular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		};
		for (const auto &model : models.models)
			for (const auto &image : model.images)
				imageDescriptors.push_back(image.descriptor);
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]),					// FS Position+Depth
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageDescriptors[1]),					// FS Normals
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &textures.whiteNoiseMap.descriptor),		// FS SSAO Noise
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &textures.blueNoiseMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, &imageDescriptors[2]),		// FS SSAO Noise
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5, &imageDescriptors[3]),		// FS SSAO Noise
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6, &textures.envMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7, &textures.envHDRCache.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8, &imageDescriptors[4]),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9, &imageDescriptors[5]),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10, &uniformBuffers.composition.descriptor),		// FS SSAO Params UBO
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 11, &imageDescriptors[6]),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12, &imageDescriptors[7]),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 13, &imageDescriptors[8], numImages),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14, &storageBuffers.sceneDesc.descriptor),
			//vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 15, &storageBuffers.envAccel.descriptor),
			//vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16, &textures.envPdfs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 17, &storageBuffers.reservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 18, &storageBuffers.previousReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 19, &storageBuffers.indirectReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20, &storageBuffers.previousIndirectReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 21, &storageBuffers.incidentRadianceGrid.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 22, &storageBuffers.gmmStatisticsPack0.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 23, &storageBuffers.gmmStatisticsPack1.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24, &storageBuffers.gmmStatisticsPack0Prev.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 25, &storageBuffers.gmmStatisticsPack1Prev.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 26, &storageBuffers.vpls.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.rayTracing, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 28, &storageBuffers.boundingVoxels.descriptor),
		};
		VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;

		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet = descriptorSets.rayTracing;
		accelerationStructureWrite.dstBinding = 27;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		writeDescriptorSets.push_back(accelerationStructureWrite);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		//Blur
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Sampler SSAO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 5),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 6),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 7),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 8),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 9),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 10),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 11),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 12),
		};

		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.blur));
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.blur;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.blur));
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.blur;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.blur));
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.rayTracing.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.rayTracing.history.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		};
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]), //samplerColor
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageDescriptors[1]), //samplerPosition
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageDescriptors[2]), //samplerNormal
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &imageDescriptors[3]), //samplerHistory
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, &uniformBuffers.blur.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &storageBuffers.reservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &storageBuffers.previousReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &storageBuffers.indirectReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8, &storageBuffers.previousIndirectReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9, &storageBuffers.gmmStatisticsPack0.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10, &storageBuffers.gmmStatisticsPack1.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11, &storageBuffers.gmmStatisticsPack0Prev.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.blur, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12, &storageBuffers.gmmStatisticsPack1Prev.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// Composition
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Position+Depth
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),						// FS Normals
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),						// FS Albedo
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),						// FS Lighting
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 5),						// FS SSAO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 6),						// FS SSAO blurred
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 7),						// FS EnvMap
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 8),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 9),							// FS Lights UBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 10),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 11),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 12),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 13),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 14),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 15),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 16),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 17),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 18),
		};
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.composition));
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.composition;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.composition));
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.composition;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.composition));
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.specular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.rayTracing.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.blur.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.motion.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.rayTracing.history.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		};
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]),			// FS Sampler Position+Depth
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageDescriptors[1]),			// FS Sampler Normals
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageDescriptors[2]),			// FS Sampler Albedo
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &imageDescriptors[3]),			// FS Sampler Specular
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, &imageDescriptors[4]),			// FS Sampler SSAO
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5, &imageDescriptors[5]),			// FS Sampler SSAO blurred
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6, &textures.envMap.descriptor),	// FS Sampler EnvMap
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7, &imageDescriptors[6]),			
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8, &imageDescriptors[7]),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 9, &uniformBuffers.blur.descriptor),	// SSAO Params UBO
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10, &uniformBuffers.composition.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11, &storageBuffers.reservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12, &storageBuffers.indirectReservoirs.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13, &storageBuffers.vpls.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14, &storageBuffers.gmmStatisticsPack0.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 15, &storageBuffers.gmmStatisticsPack1.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16, &storageBuffers.gmmStatisticsPack0Prev.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 17, &storageBuffers.gmmStatisticsPack1Prev.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 18, &storageBuffers.screen.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	void preparePipelines()
	{
		specializationData.incidentRadianceMapSize = incidentRadianceMapSize;
		std::array<VkSpecializationMapEntry, 33> specializationMapEntries = {
			vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, textured), sizeof(SpecializationData::textured)),
			vks::initializers::specializationMapEntry(1, offsetof(SpecializationData, bumped), sizeof(SpecializationData::bumped)),
			vks::initializers::specializationMapEntry(2, offsetof(SpecializationData, orthogonalize), sizeof(SpecializationData::orthogonalize)),
			vks::initializers::specializationMapEntry(3, offsetof(SpecializationData, gammaCorrection), sizeof(SpecializationData::gammaCorrection)),
			vks::initializers::specializationMapEntry(4, offsetof(SpecializationData, alphaTest), sizeof(SpecializationData::alphaTest)),
			vks::initializers::specializationMapEntry(5, offsetof(SpecializationData, toneMapping), sizeof(SpecializationData::toneMapping)),
			vks::initializers::specializationMapEntry(6, offsetof(SpecializationData, animateNoise), sizeof(SpecializationData::animateNoise)),
			vks::initializers::specializationMapEntry(7, offsetof(SpecializationData, nee), sizeof(SpecializationData::nee)),
			vks::initializers::specializationMapEntry(8, offsetof(SpecializationData, restirDI), sizeof(SpecializationData::restirDI)),
			vks::initializers::specializationMapEntry(9, offsetof(SpecializationData, envMapIS), sizeof(SpecializationData::envMapIS)),
			vks::initializers::specializationMapEntry(10, offsetof(SpecializationData, visibilityReuse), sizeof(SpecializationData::visibilityReuse)),
			vks::initializers::specializationMapEntry(11, offsetof(SpecializationData, temporalReuseDI), sizeof(SpecializationData::temporalReuseDI)),
			vks::initializers::specializationMapEntry(12, offsetof(SpecializationData, temporalReuseGI), sizeof(SpecializationData::temporalReuseGI)),
			vks::initializers::specializationMapEntry(13, offsetof(SpecializationData, spatialReuseDI), sizeof(SpecializationData::spatialReuseDI)),
			vks::initializers::specializationMapEntry(14, offsetof(SpecializationData, spatialReuseGI), sizeof(SpecializationData::spatialReuseGI)),
			vks::initializers::specializationMapEntry(15, offsetof(SpecializationData, restirGI), sizeof(SpecializationData::restirGI)),
			vks::initializers::specializationMapEntry(16, offsetof(SpecializationData, debug), sizeof(SpecializationData::debug)),
			vks::initializers::specializationMapEntry(17, offsetof(SpecializationData, spotLight), sizeof(SpecializationData::spotLight)),
			vks::initializers::specializationMapEntry(18, offsetof(SpecializationData, environmentMap), sizeof(SpecializationData::environmentMap)),
			vks::initializers::specializationMapEntry(19, offsetof(SpecializationData, biasCorrectionDI), sizeof(SpecializationData::biasCorrectionDI)),
			vks::initializers::specializationMapEntry(20, offsetof(SpecializationData, biasCorrectionGI), sizeof(SpecializationData::biasCorrectionGI)),
			vks::initializers::specializationMapEntry(21, offsetof(SpecializationData, geometricSimilarityDI), sizeof(SpecializationData::geometricSimilarityDI)),
			vks::initializers::specializationMapEntry(22, offsetof(SpecializationData, geometricSimilarityGI), sizeof(SpecializationData::geometricSimilarityGI)),
			vks::initializers::specializationMapEntry(23, offsetof(SpecializationData, jacobian), sizeof(SpecializationData::jacobian)),
			vks::initializers::specializationMapEntry(24, offsetof(SpecializationData, incidentRadianceMapSize), sizeof(SpecializationData::incidentRadianceMapSize)),
			vks::initializers::specializationMapEntry(25, offsetof(SpecializationData, pathGuiding), sizeof(SpecializationData::pathGuiding)),
			vks::initializers::specializationMapEntry(26, offsetof(SpecializationData, neeMIS), sizeof(SpecializationData::neeMIS)),
			vks::initializers::specializationMapEntry(27, offsetof(SpecializationData, guidingMIS), sizeof(SpecializationData::guidingMIS)),
			vks::initializers::specializationMapEntry(28, offsetof(SpecializationData, hashing), sizeof(SpecializationData::hashing)),
			vks::initializers::specializationMapEntry(29, offsetof(SpecializationData, cdf), sizeof(SpecializationData::cdf)),
			vks::initializers::specializationMapEntry(30, offsetof(SpecializationData, sspg), sizeof(SpecializationData::sspg)),
			vks::initializers::specializationMapEntry(31, offsetof(SpecializationData, sgm), sizeof(SpecializationData::sgm)),
			vks::initializers::specializationMapEntry(32, offsetof(SpecializationData, vxpg), sizeof(SpecializationData::vxpg)),
		};
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(specializationMapEntries.size(), specializationMapEntries.data(), sizeof(specializationData), &specializationData);

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayouts.composition, renderPass, 0);
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Empty vertex input state for fullscreen passes
		VkPipelineVertexInputStateCreateInfo emptyVertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyVertexInputState;
		rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;

		// Final composition pipeline
		shaderStages[0] = loadShader("./../shaders/composition.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader("./../shaders/composition.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		shaderStages[0].pSpecializationInfo = &specializationInfo;
		shaderStages[1].pSpecializationInfo = &specializationInfo;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.composition));

		// SSAO generation pipeline (Ray Tracing with NRD outputs)
		{
			pipelineCreateInfo.renderPass = frameBuffers.rayTracing.renderPass;
			pipelineCreateInfo.layout = pipelineLayouts.rayTracing;
			// 6个color attachments: color, history, radianceHitDist, normalRoughness, motionVector, viewZ
			std::array<VkPipelineColorBlendAttachmentState, 6> blendAttachmentStates;
			for (auto i = 0; i < blendAttachmentStates.size(); ++i)
				blendAttachmentStates[i] = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
			colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
			colorBlendState.pAttachments = blendAttachmentStates.data();
			shaderStages[1] = loadShader("./../shaders/pathtracing.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			shaderStages[1].pSpecializationInfo = &specializationInfo;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.rayTracing));
		}

		// SSAO blur pipeline
		{
			pipelineCreateInfo.renderPass = frameBuffers.blur.renderPass;
			pipelineCreateInfo.layout = pipelineLayouts.blur;
			std::array<VkPipelineColorBlendAttachmentState, 4> blendAttachmentStates;
			for (auto i = 0; i < blendAttachmentStates.size(); ++i)
				blendAttachmentStates[i] = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
			colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
			colorBlendState.pAttachments = blendAttachmentStates.data();
			shaderStages[0] = loadShader("./../shaders/fullscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader("./../shaders/blur.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			shaderStages[0].pSpecializationInfo = &specializationInfo;
			shaderStages[1].pSpecializationInfo = &specializationInfo;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.blur));
		}

		// Fill G-Buffer pipeline
		{
			// Vertex input state from glTF model loader
			pipelineCreateInfo.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Tangent });
			pipelineCreateInfo.renderPass = frameBuffers.gBuffer.renderPass;
			pipelineCreateInfo.layout = pipelineLayouts.gBuffer;
			// Blend attachment states required for all color attachments
			// This is important, as color write mask will otherwise be 0x0 and you
			// won't see anything rendered to the attachment
			std::array<VkPipelineColorBlendAttachmentState, 5> blendAttachmentStates = {
				vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
				vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
				vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
				vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
				vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE)
			};
			colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
			colorBlendState.pAttachments = blendAttachmentStates.data();
			rasterizationState.cullMode = VK_CULL_MODE_NONE;
			shaderStages[0] = loadShader("./../shaders/gbuffer.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader("./../shaders/gbuffer.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			shaderStages[0].pSpecializationInfo = &specializationInfo;
			shaderStages[1].pSpecializationInfo = &specializationInfo;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.gBuffer));
		}

		// Create compute shader pipelines
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

		auto createComputePipeline = [&](VkPipeline &pipeline, const std::string &name)
		{
			computePipelineCreateInfo.stage = loadShader("./../shaders/" + name + ".comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
			computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));
		};

		if (specializationData.sspg == 0)
		{
			if (specializationData.cdf == 0)
				createComputePipeline(compute.pipelines.prepare, "prepare");
			else
				createComputePipeline(compute.pipelines.prepare, "preparecdf");
		}
		else
		{
			if (specializationData.sgm == 1)
				createComputePipeline(compute.pipelines.prepare, "learnsgm");
			else
				createComputePipeline(compute.pipelines.prepare, "learngmm");
		}

		createComputePipeline(compute.pipelines.reset, "reset");
		
		// VXPG compute pipelines
		createComputePipeline(compute.pipelines.resetVXPG, "resetvxpg");
		createComputePipeline(compute.pipelines.prepareVXPG, "preparevxpg");
	}

	void prepareCompute()
	{
		updateBuffer(compute.uniformBuffer, sizeof(compute.ubo), &compute.ubo);

		// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer)
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.pNext = NULL;
		queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		queueCreateInfo.queueCount = 1;
		vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);

		VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));
		std::vector<VkDescriptorImageInfo> imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.gBuffer.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		};
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &compute.uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &storageBuffers.incidentRadianceGrid.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &storageBuffers.gmmStatisticsPack0Prev.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &storageBuffers.gmmStatisticsPack1Prev.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &storageBuffers.vpls.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5, &imageDescriptors[0]),					// FS Position+Depth
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6, &imageDescriptors[1]),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &storageBuffers.boundingVoxels.descriptor),
		};
		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
		
		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 3);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffers[0]));

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		createBuffer(compute.uniformBuffer, sizeof(compute.ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		createBuffer(uniformBuffers.gBuffer, sizeof(uboGBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		createBuffer(uniformBuffers.composition, sizeof(uboComposition), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		createBuffer(uniformBuffers.blur, sizeof(uboBlur), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		// Update
		updateUniformBufferBlur();
	}

	void updateUniformBufferGBuffer()
	{
		uboGBuffer.previousProjection = uboGBuffer.projection;
		uboGBuffer.previousView = uboGBuffer.view;
		uboGBuffer.projection = camera.matrices.perspective;
		uboGBuffer.view = camera.matrices.view;
		uboGBuffer.model = glm::mat4(1.0f);
		uboGBuffer.viewPos = -camera.position;

		updateBuffer(uniformBuffers.gBuffer, sizeof(uboGBuffer), &uboGBuffer);
	}

	void updateUniformBufferComposition()
	{
		auto view = camera.matrices.view;
		view[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		uboComposition.inverseMVP = glm::inverse(camera.matrices.perspective * view);
		uboComposition.projection = camera.matrices.perspective;
		// Current view position
		//为什么xz要取反？
		uboComposition.viewPos = -camera.position;
		uboComposition.envRot = envRot;

		updateBuffer(uniformBuffers.composition, sizeof(uboComposition), &uboComposition);
	}

	void updateUniformBufferBlur()
	{
		updateBuffer(uniformBuffers.blur, sizeof(uboBlur), &uboBlur);
	}

	void draw()
	{
		// 如果使用NRD，需要每帧重建command buffers以更新frameIndex
		if (useNRD)
		{
			buildCommandBuffers();
		}
		
		// Wait for rendering finished
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		if (specializationData.pathGuiding || specializationData.vxpg)
		{
			computeSubmitInfo.commandBufferCount = 1;
			//也可以else用空command buffer
			computeSubmitInfo.pCommandBuffers = &compute.commandBuffers[currentBuffer];
		}
		computeSubmitInfo.waitSemaphoreCount = 1;
		computeSubmitInfo.pWaitSemaphores = &semaphore;
		computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::prepareFrame();

		VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
		VkSemaphore graphicsSignalSemaphores[] = { semaphore, semaphores.renderComplete };

		// Submit graphics commands
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
		submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		//static bool first = true;
		//if (first && specializationData.pathGuiding)
		//{
		//	std::vector<IncidentRadianceGridCell> incidentRadianceGrid(pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z);
		//	getBuffer(incidentRadianceGrid.data(), storageBuffers.incidentRadianceGrid, incidentRadianceGrid.size() * sizeof(IncidentRadianceGridCell));
		//	auto fp = fopen("incidentRadianceGrid.txt", "w");
		//	for (auto &grid : incidentRadianceGrid)
		//	{
		//		auto printed = false;
		//		for (auto i = 0; i < incidentRadianceMapSize * incidentRadianceMapSize; ++i)
		//			if (grid.incidentRadianceCount[i] > 0 || grid.incidentRadianceSum[i] > 0.0f)
		//				printed = true;
		//		if (printed)
		//		{
		//			for (auto i = 0; i < incidentRadianceMapSize * incidentRadianceMapSize; ++i)
		//				fprintf(fp, "(%d %e) ", grid.incidentRadianceCount[i], grid.incidentRadianceSum[i]);
		//			fprintf(fp, "\n");
		//		}
		//	}
		//	fclose(fp);
		//	first = false;
		//}

		// Read query results for displaying in next frame
		getQueryResults();
		auto queryIndex = (int)currentBuffer - (int)swapChain.imageCount + 1;
		if (queryIndex < 0)
			queryIndex += swapChain.imageCount;
		static auto counter = 0;
		++counter;
		if (counter >= swapChain.imageCount)
		{
			for (auto i = 0; i < 5; ++i)
				sumTimings[i] += (timestamps[queryIndex * numTimestamps + i * 2 + 1] - timestamps[queryIndex * numTimestamps + i * 2]) * timeStampPeriod;
			++timingFrameCounter;
			if (timingFrameCounter == 100)
			{
				for (auto i = 0; i < 5; ++i)
				{
					timings[i] = sumTimings[i] / 100.0;
					sumTimings[i] = 0.0;
				}
				timingFrameCounter = 0;
			}
			for (auto i = 0; i < 2; ++i)
				sumTimings[5 + i] += (computeTimestamps[queryIndex * numComputeTimestamps + i * 2 + 1] - computeTimestamps[queryIndex * numComputeTimestamps + i * 2]) * timeStampPeriod;
			++computeTimingFrameCounter;
			if (computeTimingFrameCounter == 100)
			{
				for (auto i = 5; i < 10; ++i)
				{
					timings[i] = sumTimings[i] / 100.0;
					sumTimings[i] = 0.0;
				}
				computeTimingFrameCounter = 0;
			}
		}
		timings[7] = (computeTimestamps[5] - computeTimestamps[4]) * timeStampPeriod;
		//printf("G-Buffer %fms\n", (timestamps[1] - timestamps[0]) / 1e6);
		//printf("Ray Tracing %fms\n", (timestamps[3] - timestamps[2]) / 1e6);
		//printf("Blur %fms\n", (timestamps[5] - timestamps[3]) / 1e6);
		//printf("Composition %fms\n", (timestamps[7] - timestamps[6]) / 1e6);

		VulkanExampleBase::submitFrame();
		
		// 每帧递增NRD frameIndex（在NRDWrapper内部管理）
		if (useNRD)
		{
			nrdDenoiser.incrementFrameIndex();
		}
	}

	// Take a screenshot from the current swapchain image
	// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
	// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependent optimal tiling format
	// Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
	void saveScreenshot()
	{
		bool supportsBlit = true;

		// Check blit support for source and destination
		VkFormatProperties formatProps;

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		vkGetPhysicalDeviceFormatProperties(physicalDevice, swapChain.colorFormat, &formatProps);
		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images
		vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Source for the copy is the last rendered swapchain image
		VkImage srcImage = swapChain.images[currentBuffer];

		// Create the linear tiled destination image to copy to and to read the memory from
		VkImageCreateInfo imageCreateCI(vks::initializers::imageCreateInfo());
		imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageCreateCI.extent.width = width;
		imageCreateCI.extent.height = height;
		imageCreateCI.extent.depth = 1;
		imageCreateCI.arrayLayers = 1;
		imageCreateCI.mipLevels = 1;
		imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateCI.tiling = VK_IMAGE_TILING_LINEAR;
		imageCreateCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		// Create the image
		VkImage dstImage;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateCI, nullptr, &dstImage));
		// Create memory to back up the image
		VkMemoryRequirements memRequirements;
		VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
		VkDeviceMemory dstImageMemory;
		vkGetImageMemoryRequirements(device, dstImage, &memRequirements);
		memAllocInfo.allocationSize = memRequirements.size;
		// Memory must be host visible to copy from
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &dstImageMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, dstImage, dstImageMemory, 0));

		// Do the actual blit from the swapchain image to our host visible destination image
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		// Transition destination image to transfer destination layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition swapchain image from present to transfer source layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
		if (supportsBlit)
		{
			// Define the region to blit (we will blit the whole swapchain image)
			VkOffset3D blitSize;
			blitSize.x = width;
			blitSize.y = height;
			blitSize.z = 1;
			VkImageBlit imageBlitRegion{};
			imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.srcSubresource.layerCount = 1;
			imageBlitRegion.srcOffsets[1] = blitSize;
			imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.dstSubresource.layerCount = 1;
			imageBlitRegion.dstOffsets[1] = blitSize;

			// Issue the blit command
			vkCmdBlitImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlitRegion,
				VK_FILTER_NEAREST);
		}
		else
		{
			// Otherwise use image copy (requires us to manually flip components)
			VkImageCopy imageCopyRegion{};
			imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.srcSubresource.layerCount = 1;
			imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.dstSubresource.layerCount = 1;
			imageCopyRegion.extent.width = width;
			imageCopyRegion.extent.height = height;
			imageCopyRegion.extent.depth = 1;

			// Issue the copy command
			vkCmdCopyImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageCopyRegion);
		}

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition back the swap chain image after the blit is done
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		vulkanDevice->flushCommandBuffer(copyCmd, queue);

		// Get layout of the image (including row pitch)
		VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

		// Map image memory so we can start copying from it
		const char* data;
		vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
		data += subResourceLayout.offset;

		cv::Mat screenshot(height, width, CV_8UC4, (void *)data);
		cv::cvtColor(screenshot, screenshot, cv::COLOR_RGBA2BGR);
		cv::imwrite("screenshot.png", screenshot);

		// Clean up resources
		vkUnmapMemory(device, dstImageMemory);
		vkFreeMemory(device, dstImageMemory, nullptr);
		vkDestroyImage(device, dstImage, nullptr);
	}

	void prepare() override
	{
		VulkanRaytracingSample::prepare();
		timestamps.resize(numTimestamps * swapChain.imageCount);
		computeTimestamps.resize(numComputeTimestamps * swapChain.imageCount);
		loadAssets();
		setupQueryPool();
		prepareOffscreenFramebuffers();
		prepareUniformBuffers();
		bottomLevelASList.resize(models.models.size());
		for (auto i = 0; i < models.models.size(); ++i)
			createBottomLevelAccelerationStructure(bottomLevelASList[i], models.models[i],
				VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
		createTopLevelAccelerationStructure(VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
			VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
		setupDescriptorPool();
		prepareCompute();
		setupLayoutsAndDescriptors();
		if (settings.scene == 0)
			updateView(0);
		else if (settings.scene == 1)
			updateView(3);
		else if (settings.scene == 2)
			updateView(4);
		preparePipelines();
		buildCommandBuffers();
		buildComputeCommandBuffers();
		
		// 初始化NRD Denoiser
		std::cout << "[NRD Integration] Initializing NRD denoiser..." << std::endl;
		if (nrdDenoiser.initialize(device, physicalDevice, width, height))
		{
			std::cout << "[NRD Integration] NRD denoiser initialized successfully!" << std::endl;
			std::cout << "[NRD Integration] NRD can work with Path Guiding and VXPG independently" << std::endl;
			std::cout << "[NRD Integration] Toggle 'Use NRD' in Denoiser panel to enable" << std::endl;
		}
		else
		{
			std::cerr << "[NRD Integration] Failed to initialize NRD denoiser!" << std::endl;
		}
		
		prepared = true;

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore));

		// Signal the semaphore
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(queue));
	}

	// Debug: Print VXPG voxel data to check if it's learning
	void debugPrintVoxelData()
	{
		if (specializationData.vxpg == 0) {
			printf("[VXPG Debug] VXPG is not enabled\n");
			return;
		}

		// Create staging buffer if not exists
		VkDeviceSize bufferSize = pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z * sizeof(BoundingVoxel);
		if (storageBuffers.voxelDebugStaging.buffer == VK_NULL_HANDLE)
		{
			createBuffer(storageBuffers.voxelDebugStaging, bufferSize, 
				VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		}

		// Copy data from device to staging buffer
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(copyCmd, storageBuffers.boundingVoxels.buffer, storageBuffers.voxelDebugStaging.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue);

		// Map and read data
		BoundingVoxel* voxels = nullptr;
		VK_CHECK_RESULT(vkMapMemory(device, storageBuffers.voxelDebugStaging.memory, 0, bufferSize, 0, (void**)&voxels));

		// Statistics
		int totalVoxels = pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z;
		int nonZeroIrradiance = 0;
		int nonZeroSampleCount = 0;
		float totalIrradiance = 0.0f;
		int totalSamples = 0;
		float maxIrradiance = 0.0f;
		int maxSamples = 0;

		for (int i = 0; i < totalVoxels; i++) {
			if (voxels[i].totalIrradiance > 0.0f) {
				nonZeroIrradiance++;
				totalIrradiance += voxels[i].totalIrradiance;
				if (voxels[i].totalIrradiance > maxIrradiance)
					maxIrradiance = voxels[i].totalIrradiance;
			}
			if (voxels[i].sampleCount > 0) {
				nonZeroSampleCount++;
				totalSamples += voxels[i].sampleCount;
				if (voxels[i].sampleCount > maxSamples)
					maxSamples = voxels[i].sampleCount;
			}
		}

		// Open file for writing (append mode)
		FILE* file = fopen("vxpg_debug.txt", "a");
		if (!file) {
			printf("[VXPG Debug] Failed to open vxpg_debug.txt for writing\n");
			vkUnmapMemory(device, storageBuffers.voxelDebugStaging.memory);
			return;
		}

		// Write to file instead of console
		fprintf(file, "\n========== VXPG Debug Info (Frame %d) ==========\n", frameNumber);
		fprintf(file, "Grid: %d x %d x %d = %d voxels\n", 
			pushConstants.gridDim.x, pushConstants.gridDim.y, pushConstants.gridDim.z, totalVoxels);
		fprintf(file, "Voxels with irradiance > 0: %d / %d (%.1f%%)\n", 
			nonZeroIrradiance, totalVoxels, 100.0f * nonZeroIrradiance / totalVoxels);
		fprintf(file, "Voxels with samples > 0: %d / %d (%.1f%%)\n", 
			nonZeroSampleCount, totalVoxels, 100.0f * nonZeroSampleCount / totalVoxels);
		fprintf(file, "Total irradiance: %.6f\n", totalIrradiance);
		fprintf(file, "Total samples: %d\n", totalSamples);
		fprintf(file, "Max irradiance: %.6f\n", maxIrradiance);
		fprintf(file, "Max samples: %d\n", maxSamples);
		if (nonZeroIrradiance > 0)
			fprintf(file, "Avg irradiance (non-zero): %.6f\n", totalIrradiance / nonZeroIrradiance);
		if (nonZeroSampleCount > 0)
			fprintf(file, "Avg samples (non-zero): %.1f\n", (float)totalSamples / nonZeroSampleCount);

		// Print first 20 voxels with data
		fprintf(file, "\nFirst 20 voxels with irradiance:\n");
		int printed = 0;
		for (int i = 0; i < totalVoxels && printed < 20; i++) {
			if (voxels[i].totalIrradiance > 0.0f || voxels[i].sampleCount > 0) {
				fprintf(file, "  Voxel[%d]: irradiance=%.6f, samples=%d\n", 
					i, voxels[i].totalIrradiance, voxels[i].sampleCount);
				printed++;
			}
		}
		fprintf(file, "================================================\n\n");

		fclose(file);
		
		// Also print a brief summary to console
		printf("[VXPG Debug Frame %d] Saved to vxpg_debug.txt - Irradiance: %.3f, Samples: %d, Active voxels: %.1f%%\n",
			frameNumber, totalIrradiance, totalSamples, 100.0f * nonZeroSampleCount / totalVoxels);

		vkUnmapMemory(device, storageBuffers.voxelDebugStaging.memory);
	}

	void render() override
	{
		if (!prepared)
			return;

		if (specializationData.animateNoise)
			uboComposition.frameNumber = frameNumber;

		updateUniformBufferComposition();
		updateUniformBufferGBuffer();

		draw();
		
		// Debug: Print VXPG data every 100 frames
		if (specializationData.vxpg && frameNumber > 0 && frameNumber % 100 == 0)
		{
			debugPrintVoxelData();
		}
		
		if (uboComposition.screenshot)
		{
			uboComposition.screenshot = false;
			cv::Mat screen(height, width, CV_32FC3);
			getBuffer(screen.data, storageBuffers.screen, width * height * sizeof(glm::vec3));
			cv::imwrite("screenshot.exr", screen);
		}
	}

	void windowResized() override
	{
		//frameBuffers.gBuffer.position.destroy(device);
		//frameBuffers.gBuffer.normal.destroy(device);
		//frameBuffers.gBuffer.albedo.destroy(device);
		//frameBuffers.gBuffer.depth.destroy(device);
		//frameBuffers.rayTracing.color.destroy(device);
		//frameBuffers.blur.color.destroy(device);

		//frameBuffers.gBuffer.destroy(device);
		//frameBuffers.rayTracing.destroy(device);
		//frameBuffers.blur.destroy(device);

		//prepareOffscreenFramebuffers();
		//preparePipelines();
		//buildCommandBuffers();

		//resized = false;
	}

	void viewChanged() override
	{
		if (uboComposition.reference)
			frameNumber = 0;
		if (envUpdated)
			envUpdated = false;
	}

	void updateView(int viewIndex)
	{
		if (viewIndex == 0)
		{
			//camera.position = { -5.5238f, -6.6096f, 0.6805f };
			//camera.setRotation({ 13.068653f, -105.455933f, 0.0f });
			//uboComposition.exposure = 2.0f;
			camera.position = { -1.919526f, -1.531252f, -3.439754f };
			camera.setRotation({ 0.918655f, -95.330978f, 0.0f });
			uboComposition.exposure = 10.0f;
			uboBlur.blur = false;
			uboComposition.numBounces = 100;
			specializationData.nee = false;
			specializationData.restirDI = false;
			uboComposition.sampleCount = 1;
			uboComposition.historyLength = 20;
			specializationData.restirGI = false;
			specializationData.textured = true;
			uboComposition.spotLightPos = glm::vec3(-1.7965f, -4.2661f, -1.7335f);
			uboComposition.spotDir = glm::normalize(glm::vec3(0.0f, -1.0f, 1.0f));
			updateUniformBufferBlur();
		}
		if (viewIndex == 3)
		{
			//animation interior
			camera.position = { 6.389852f, -2.090485f, -10.561940f };
			camera.setRotation({ 0.0f, 116.112869f, 0.0f });
			//animation exterior
			camera.position = { 20.678839, -3.309801, -9.261426 };
			camera.setRotation({ 0.0f, -7.243255, 0.0f });
			//bistro interior
			camera.position = { 11.285220f, -2.784755f, -8.135830f };
			camera.setRotation({ 12.393642f, 116.112869f, 0.0f });
			////onlyVisible
			//camera.position = { 9.374372f, -2.605895f, -11.613957f };
			//camera.setRotation({ 2.606144f, 61.100254f, 0.0f });
			////portalWeighted
			//camera.position = { 4.970885f, -3.526999f, -9.592390f };
			//camera.setRotation({ 23.699900f, 168.762726f, 0.0f });
			////mis
			//camera.position = { 8.948537f, -1.752038f, -9.041840f };
			//camera.setRotation({ 10.706150f, 298.700104f, 0.0f });
			////onlyVisible
			//camera.position = { 13.457746f, -1.631146f, -14.067551f };
			//camera.setRotation({ 19.987434f, 506.600647f, 0.0f });
			////radianceWeighted
			//camera.position = { 5.265397f, -3.448425f, -9.187300f };
			//camera.setRotation({ 19.987440f, 298.025513f, 0.0f });
			////bistro exterior
			//camera.position = { 13.912436f, -3.994066f, -23.186253f };
			//camera.setRotation({ 5.981144f, 341.056763f, 0.0f });
			//bistro exterior 2
			//camera.position = { 26.605881f, -6.050263f, -10.660443f };
			//camera.setRotation({ 13.743652f, 92.150307f, 0.0f });
			uboComposition.exposure = 50.0f;
			//uboComposition.exposure = 30.0f;
			//uboComposition.exposure = 6.0f;
			specializationData.textured = true;
			uboBlur.blur = false;
			uboComposition.numBounces = 100;
			specializationData.nee = false;
			specializationData.pathGuiding = false;
			specializationData.restirDI = false;
			uboComposition.sampleCount = 1;
			uboComposition.historyLength = 20;
			specializationData.restirGI = false;
			specializationData.spatialReuseGI = false;
			specializationData.environmentMap = true;
			//specializationData.environmentMap = false;
			//specializationData.spotLight = true;
			specializationData.spotLight = false;
			//exterior
			uboComposition.spotLightPos = glm::vec3(-16.704531f, 6.587162f, 15.237517f);
			uboComposition.spotDir = glm::normalize(glm::vec3(0.0f, -1.0f, 1.0f));
			//interior
			//uboComposition.spotLightPos = glm::vec3(-8.929632f, 3.381073f, 11.737566f);
			//uboComposition.spotDir = glm::normalize(glm::vec3(1.0f, -1.0f, -1.0f));
			uboComposition.spotAngle = 80.0f;
			uboComposition.spotLightIntensity = 400.0f;
			updateUniformBufferBlur();
		}
		if (viewIndex == 4)
		{
			//sanmiguel inside
			camera.position = { -25.015345f, -5.110816f, -13.057541f };
			camera.setRotation({ 36.205399f, 1.506813f, 0.0f });
			//onlyVisible
			//camera.position = { -25.015345f, -5.110816f, -13.057541f };
			//camera.setRotation({ 1.931134f, -5.387186f, 0.0f });
			//portalWeighted
			//camera.position = { -25.462612f, -4.639282, -13.428738f };
			//camera.setRotation({ 21.186626f, -28.024439f, 0.0f });
			//radianceWeighted
			//camera.position = { -24.234716f, -4.389646f, -13.008386f };
			//camera.setRotation({ 22.874125f, 20.069326f, 0.0f });
			//sanmiguel outside
			//camera.position = { -9.030672f, -18.604053f, -5.243932f };
			//camera.setRotation({ 51.898968f, 87.738159f, 0.0f });
			uboComposition.exposure = 20.0f;
			specializationData.textured = true;
			uboBlur.blur = false;
			uboComposition.numBounces = 100;
			specializationData.nee = false;
			specializationData.restirDI = false;
			uboComposition.sampleCount = 1;
			uboComposition.historyLength = 20;
			specializationData.restirGI = false;
			specializationData.spatialReuseGI = false;
			specializationData.bumped = false;
			specializationData.environmentMap = true;
			//specializationData.environmentMap = false;
			//specializationData.spotLight = true;
			specializationData.spotLight = false;
			//inside
			//uboComposition.spotLightPos = glm::vec3(24.146687f, 3.776174f, 11.781631f);
			//outside
			uboComposition.spotLightPos = glm::vec3(22.057491f, 2.382178f, -1.862019f);
			uboComposition.spotDir = glm::normalize(glm::vec3(1.0f, -1.0f, 0.0f));
			uboComposition.spotAngle = 80.0f;
			uboComposition.spotLightIntensity = 400.0f;
			updateUniformBufferBlur();
		}
		viewChanged();
	}

	void reset()
	{
		auto commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		auto commandBufferIndex = 0;

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
		vkCmdPushConstants(commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);
		vkCmdResetQueryPool(commandBuffer, computeQueryPool, 4, 2);

		auto dispatchCompute = [this, &commandBuffer, &commandBufferIndex](const VkPipeline &pipeline, glm::uvec3 groupCount, int timeStampOffset = -1)
		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			if (timeStampOffset >= 0)
				vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, computeQueryPool, commandBufferIndex * numComputeTimestamps + timeStampOffset);
			vkCmdDispatch(commandBuffer, groupCount.x, groupCount.y, groupCount.z);
			if (timeStampOffset >= 0)
				vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, computeQueryPool, commandBufferIndex * numComputeTimestamps + timeStampOffset + 1);
		};

		dispatchCompute(compute.pipelines.reset, glm::uvec3(pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z, 1, 1), 4);
		
		// Reset VXPG buffers
		if (specializationData.vxpg == 1)
		{
			dispatchCompute(compute.pipelines.resetVXPG, glm::uvec3((pushConstants.gridDim.x * pushConstants.gridDim.y * pushConstants.gridDim.z + 63u) / 64, 1, 1), 6);
		}

		vulkanDevice->flushCommandBuffer(commandBuffer, queue);
	}
	
	void OnUpdateUIOverlay(vks::UIOverlay *overlay) override
	{
		if (overlay->header("Settings")) {
			overlay->comboBox("Display", &uboComposition.debugDisplayTarget,
				{ "Final Composition", //0
				"Position", //1
				"Normals", //2
				"Albedo", //3
				"Sample Seed", //4
				"Depth", //5
				"UV", //6
				"Indirect", //7
				"Indirect Blur", //8
				"Lo", //9
				"W", //10
				"M", //11
				"Num Cones", //12
				"Motion Vector", //13
				"History", //14
				"Depth Comparison", //15
				"VPL Position", //16
				"VPL Luminance", //17
				"GMM", //18
				});
			overlay->sliderInt("Num Bounces", &uboComposition.numBounces, 1, 100);
			overlay->text(std::to_string(frameNumber).c_str());
			if (overlay->checkBox("Reference", &uboComposition.reference))
				if (uboComposition.reference)
					frameNumber = 0;
			if (overlay->button("Screenshot"))
			{
				uboComposition.screenshot = true;
				saveScreenshot();
			}
			if (overlay->checkBox("Environment Map", &specializationData.environmentMap))
				preparePipelines();
			if (overlay->checkBox("Russian Roulette", &uboComposition.russianRoulette))
				preparePipelines();
			if (overlay->sliderInt("Debug", &specializationData.debug, 0, 10))
				preparePipelines();
			overlay->sliderFloat("Clamp Value", &uboComposition.clampValue, 10.0f, 200.0f);
		}
		if (overlay->header("Denoiser"))
		{
			// NRD Denoiser控制
			bool prevUseNRD = useNRD;
			overlay->checkBox("Use NRD", &useNRD);
			// 检测状态改变，重置frameIndex
			if (useNRD && !prevUseNRD)
			{
				nrdFrameIndex = 0;
				nrdDenoiser.resetFrameIndex();  // 重置NRD内部的frame counter
				std::cout << "[NRD] Enabled - Reset internal frame index" << std::endl;
			}
			else if (!useNRD && prevUseNRD)
			{
				std::cout << "[NRD] Disabled" << std::endl;
			}
			
			if (useNRD)
			{
				overlay->text("NRD Status: Enabled");
				overlay->text("(Works with Path Guiding & VXPG)");
			}
			
			overlay->sliderInt("History Length", &uboComposition.historyLength, 1, 50);
			if (overlay->checkBox("Blur", &uboBlur.blur))
				updateUniformBufferBlur();
			if (overlay->sliderInt("Blur Radius", &uboBlur.radius, 1, 20))
				updateUniformBufferBlur();
			//if (overlay->sliderFloat("BlurDepthSharpness", &uboBlur.depthSharpness, 0.0f, 5000.0f))
			if (overlay->sliderFloat("Blur Depth Sharpness", &uboBlur.depthSharpness, 0.001f, 0.1f))
				updateUniformBufferBlur();
			//if (overlay->sliderFloat("BlurNormalSharpness", &uboBlur.normalSharpness, 0.0f, 100.0f))
			if (overlay->sliderFloat("Blur Normal Sharpness", &uboBlur.normalSharpness, 0.1f, 50.0f))
				updateUniformBufferBlur();
		}
		if (overlay->header("Sampling"))
		{
			int sampleCountIndex = log(uboComposition.sampleCount) / log(2);
			if (overlay->comboBox("Sample Count", &sampleCountIndex, { "1", "2", "4", "8", "16", "32", "64", "128", "256", "512" }))
			{
				uboComposition.sampleCount = pow(2, sampleCountIndex);
				preparePipelines();
			}
			if (overlay->checkBox("NEE", &specializationData.nee))
				preparePipelines();
			if (overlay->checkBox("NEE MIS", &specializationData.neeMIS))
				preparePipelines();
			if (overlay->checkBox("Env Map IS", &specializationData.envMapIS))
				preparePipelines();
			if (overlay->checkBox("Animate Noise", &specializationData.animateNoise))
			{
				if (specializationData.animateNoise == 1)
					frameNumber = 0;
				preparePipelines();
			}
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
			if (overlay->checkBox("VXPG", &specializationData.vxpg))
				preparePipelines();
			overlay->sliderFloat("Probability", &uboComposition.probability, 0.0f, 1.0f);
			if (overlay->checkBox("Guiding MIS", &specializationData.guidingMIS))
				preparePipelines();
		}
		if (overlay->header("ReSTIR DI"))
		{
			if (overlay->checkBox("Enable ReSTIR DI", &specializationData.restirDI))
				preparePipelines();
			overlay->sliderInt("Num Candidates", &uboComposition.numCandidates, 1, 200);
			if (overlay->checkBox("Temporal Reuse DI", &specializationData.temporalReuseDI))
				preparePipelines();
			overlay->sliderInt("Temporal Samples DI", &uboComposition.temporalSamplesDI, 1, 100);
			if (overlay->checkBox("Spatial Reuse DI", &specializationData.spatialReuseDI))
				preparePipelines();
			overlay->sliderInt("Spatial Samples DI", &uboComposition.spatialSamplesDI, 1, 10);
			overlay->sliderInt("Spatial Reuse Radius DI", &uboComposition.spatialReuseRadiusDI, 0, 64);
			if (overlay->checkBox("Visibility Reuse", &specializationData.visibilityReuse))
				preparePipelines();
			if (overlay->checkBox("Bias Correction DI", &specializationData.biasCorrectionDI))
				preparePipelines();
		}
		if (overlay->header("ReSTIR GI"))
		{
			//不能和ReSTIR DI的重名
			if (overlay->checkBox("Enable ReSTIR GI", &specializationData.restirGI))
				preparePipelines();
			if (overlay->checkBox("Temporal Reuse GI", &specializationData.temporalReuseGI))
				preparePipelines();
			overlay->sliderInt("Temporal Samples GI", &uboComposition.temporalSamplesGI, 1, 100);
			if (overlay->checkBox("Spatial Reuse GI", &specializationData.spatialReuseGI))
				preparePipelines();
			overlay->sliderInt("Spatial Samples GI", &uboComposition.spatialSamplesGI, 1, 10);
			overlay->sliderInt("Spatial Reuse Radius GI", &uboComposition.spatialReuseRadiusGI, 0, 64);
			if (overlay->checkBox("Geometric Similarity GI", &specializationData.geometricSimilarityGI))
				preparePipelines();
			if (overlay->checkBox("Jacobian", &specializationData.jacobian))
				preparePipelines();
			if (overlay->checkBox("Bias Correction GI", &specializationData.biasCorrectionGI))
				preparePipelines();
		}

		if (overlay->header("Material"))
		{
			if (overlay->checkBox("Textured", &specializationData.textured))
				preparePipelines();
			if (overlay->checkBox("Bumped", &specializationData.bumped))
				preparePipelines();
			if (overlay->checkBox("Orthogonalize", &specializationData.orthogonalize))
				preparePipelines();
			if (overlay->checkBox("Alpha Test", &specializationData.alphaTest))
				preparePipelines();
			if (overlay->checkBox("Gamma Correction", &specializationData.gammaCorrection))
				preparePipelines();
			if (overlay->sliderFloat("Alpha Cutoff", &uboGBuffer.alphaCutoff, 0.0f, 1.0f))
				uboComposition.alphaCutoff = uboGBuffer.alphaCutoff;
		}

		if (overlay->header("Post Processing"))
		{
			overlay->sliderFloat("Exposure", &uboComposition.exposure, 0.0f, 50.0f);
			if (overlay->checkBox("Tone Mapping", &specializationData.toneMapping))
				preparePipelines();
		}

		if (overlay->header("Spot Light"))
		{
			if (overlay->checkBox("Enable Spot Light", &specializationData.spotLight))
				preparePipelines();
			if (overlay->button("-x"))
			{
				uboComposition.spotLightPos.x -= 1.0f;
				frameNumber = 0;
			}
			if (overlay->button("+x"))
			{
				uboComposition.spotLightPos.x += 1.0f;
				frameNumber = 0;
			}
			if (overlay->button("-y"))
			{
				uboComposition.spotLightPos.y -= 1.0f;
				frameNumber = 0;
			}
			if (overlay->button("+y"))
			{
				uboComposition.spotLightPos.y += 1.0f;
				frameNumber = 0;
			}
			if (overlay->button("-z"))
			{
				uboComposition.spotLightPos.z -= 1.0f;
				frameNumber = 0;
			}
			if (overlay->button("+z"))
			{
				uboComposition.spotLightPos.z += 1.0f;
				frameNumber = 0;
			}
			if (overlay->sliderFloat("Angle", &uboComposition.spotAngle, 0.0f, 180.0f))
				frameNumber = 0;
			if (overlay->sliderFloat("Intensity", &uboComposition.spotLightIntensity, 0.0f, 10000.0f))
				frameNumber = 0;
			//printf("%ff, %ff, %ff\n", uboComposition.spotPos.x, uboComposition.spotPos.y, uboComposition.spotPos.z);
		}

		if (overlay->header("Probe"))
		{
			if (overlay->button("-X"))
				uboGBuffer.probePos.x -= 0.2f;
			if (overlay->button("+X"))
				uboGBuffer.probePos.x += 0.2f;
			if (overlay->button("-Y"))
				uboGBuffer.probePos.y -= 0.2f;
			if (overlay->button("+Y"))
				uboGBuffer.probePos.y += 0.2f;
			if (overlay->button("-Z"))
				uboGBuffer.probePos.z -= 0.2f;
			if (overlay->button("+Z"))
				uboGBuffer.probePos.z += 0.2f;
			// printf("%f %f %f\n", uboGBuffer.probePos.x, uboGBuffer.probePos.y, uboGBuffer.probePos.z);  // 已屏蔽probe位置打印
		}
	}
};

VULKAN_EXAMPLE_MAIN()
