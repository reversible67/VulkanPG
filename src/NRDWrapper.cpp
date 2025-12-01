#include "NRDWrapper.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

// NRD Headers
#include "../external/NRD/Include/NRD.h"
#include "../external/NRD/Include/NRDDescs.h"
#include "../external/NRD/Include/NRDSettings.h"

// Phase 1: Placeholder implementation
// This allows the project to compile and run without NRD library
// Phase 2: Replace with full NRD integration

bool NRDWrapper::initialize(VkDevice device, VkPhysicalDevice physDevice,
                           uint32_t width, uint32_t height)
{
    m_device = device;
    m_physicalDevice = physDevice;
    m_width = width;
    m_height = height;
    
    std::cout << "[NRD] Initializing wrapper (" << width << "x" << height << ")..." << std::endl;
    
    // 创建输出纹理
    createOutputTexture();
    
    // 创建历史纹理（用于简化降噪）
    createHistoryTexture();
    
    // 创建NRD Instance
    const nrd::DenoiserDesc denoiserDescs[] = {
        { nrd::Identifier(0), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR }
    };
    
    nrd::InstanceCreationDesc instanceCreationDesc = {};
    instanceCreationDesc.denoisers = denoiserDescs;
    instanceCreationDesc.denoisersNum = 1;
    
    nrd::Result result = nrd::CreateInstance(instanceCreationDesc, m_nrdInstance);
    if (result != nrd::Result::SUCCESS)
    {
        std::cerr << "[NRD] Failed to create NRD instance!" << std::endl;
        return false;
    }
    
    const nrd::InstanceDesc& instanceDesc = *nrd::GetInstanceDesc(*m_nrdInstance);
    std::cout << "[NRD] Instance created with " << instanceDesc.permanentPoolSize << " bytes permanent pool" << std::endl;
    std::cout << "[NRD] Instance created with " << instanceDesc.transientPoolSize << " bytes transient pool" << std::endl;
    
    // 创建简化降噪pipeline
    createTemporalDenoisingPipeline();
    
    // TODO: Phase 2B - 创建完整NRD compute pipelines
    
    m_initialized = true;
    std::cout << "[NRD] Wrapper initialized successfully!" << std::endl;
    std::cout << "[NRD] Using REBLUR_DIFFUSE_SPECULAR denoiser" << std::endl;
    
    return true;
}

void NRDWrapper::setInputTextures(
    VkImageView radianceHitDist,
    VkImageView normalRoughness,
    VkImageView viewZ,
    VkImageView motionVector)
{
    m_inputRadianceHitDist = radianceHitDist;
    m_inputNormalRoughness = normalRoughness;
    m_inputViewZ = viewZ;
    m_inputMotionVector = motionVector;
}

void NRDWrapper::setOutputTexture(VkImageView outputView)
{
    // 使用外部提供的output view，而不是我们自己创建的
    m_outputView = outputView;
}

void NRDWrapper::denoise(VkCommandBuffer cmd, uint32_t frameIndex,
                        const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                        const glm::mat4& prevViewMatrix, const glm::mat4& prevProjMatrix)
{
    if (!m_initialized || !m_nrdInstance)
        return;
    
    // 简化版：直接执行compute shader，跳过完整的NRD API调用
    // TODO: 之后再调试完整的NRD API参数问题
    
    // 直接执行简化降噪（如果pipeline已创建）
    if (m_temporalPipeline != VK_NULL_HANDLE && 
        m_inputRadianceHitDist != VK_NULL_HANDLE &&
        m_inputNormalRoughness != VK_NULL_HANDLE &&
        m_inputMotionVector != VK_NULL_HANDLE &&
        m_inputViewZ != VK_NULL_HANDLE)
    {
        // 只在第一次更新descriptor sets（避免在command buffer记录期间更新）
        static bool descriptorsUpdated = false;
        if (!descriptorsUpdated)
        {
            std::vector<VkWriteDescriptorSet> writes(7);
            std::vector<VkDescriptorImageInfo> imageInfos(7);
        
        // Binding 0: inRadianceHitDist
        imageInfos[0].imageView = m_inputRadianceHitDist;
        imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_temporalDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &imageInfos[0];
        
        // Binding 1: inNormalRoughness
        imageInfos[1].imageView = m_inputNormalRoughness;
        imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_temporalDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &imageInfos[1];
        
        // Binding 2: inMotionVector
        imageInfos[2].imageView = m_inputMotionVector;
        imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_temporalDescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &imageInfos[2];
        
        // Binding 3: inViewZ
        imageInfos[3].imageView = m_inputViewZ;
        imageInfos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = m_temporalDescriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].pImageInfo = &imageInfos[3];
        
        // Binding 4: inHistory
        imageInfos[4].imageView = m_historyView;
        imageInfos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = m_temporalDescriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[4].pImageInfo = &imageInfos[4];
        
        // Binding 5: outDenoised
        imageInfos[5].imageView = m_outputView;
        imageInfos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = m_temporalDescriptorSet;
        writes[5].dstBinding = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[5].pImageInfo = &imageInfos[5];
        
        // Binding 6: outHistory
        imageInfos[6].imageView = m_historyView;
        imageInfos[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = m_temporalDescriptorSet;
        writes[6].dstBinding = 6;
        writes[6].descriptorCount = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[6].pImageInfo = &imageInfos[6];
        
            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            descriptorsUpdated = true;
            
            if (frameIndex <= 1)
            {
                std::cout << "[NRD] Descriptor sets initialized successfully" << std::endl;
            }
        }
        
        // Bind pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
        
        // Bind descriptor set
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipelineLayout,
                               0, 1, &m_temporalDescriptorSet, 0, nullptr);
        
        // Push constants
        struct PushConstants {
            uint32_t frameIndex;
            float accumulationWeight;
            uint32_t resetHistory;
            uint32_t padding;
        } pushConstants;
        
        // 使用内部frameIndex（每次denoise调用时自动递增）
        pushConstants.frameIndex = m_internalFrameIndex;
        // 降低weight让效果更明显：70% history + 30% current
        pushConstants.accumulationWeight = 0.7f;
        pushConstants.resetHistory = (m_internalFrameIndex == 0) ? 1 : 0;
        pushConstants.padding = 0;
        
        // 详细调试：前10帧和每60帧显示状态
        if (m_internalFrameIndex < 10 || m_internalFrameIndex % 60 == 0)
        {
            std::cout << "[NRD] Internal frame " << m_internalFrameIndex 
                      << " | resetHistory=" << pushConstants.resetHistory
                      << " | weight=" << pushConstants.accumulationWeight << std::endl;
        }
        
        vkCmdPushConstants(cmd, m_temporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pushConstants), &pushConstants);
        
        // Dispatch
        uint32_t groupCountX = (m_width + 15) / 16;
        uint32_t groupCountY = (m_height + 15) / 16;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
        
        // Memory barrier
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        if (frameIndex % 60 == 0)
        {
            std::cout << "[NRD] Simplified temporal denoising executed" << std::endl;
            std::cout << "[NRD] Dispatch: " << groupCountX << "x" << groupCountY << " groups" << std::endl;
            std::cout << "[NRD] Output view: " << (m_outputView != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD] History view: " << (m_historyView != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
        }
        
        // 在第一帧额外输出调试信息
        if (frameIndex == 1)
        {
            std::cout << "[NRD DEBUG] First frame execution:" << std::endl;
            std::cout << "[NRD DEBUG] - Pipeline: " << (m_temporalPipeline != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Input radianceHitDist: " << (m_inputRadianceHitDist != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Input normalRoughness: " << (m_inputNormalRoughness != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Input viewZ: " << (m_inputViewZ != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Input motionVector: " << (m_inputMotionVector != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Output view: " << (m_outputView != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - History view: " << (m_historyView != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Descriptor set: " << (m_temporalDescriptorSet != VK_NULL_HANDLE ? "Valid" : "NULL") << std::endl;
            std::cout << "[NRD DEBUG] - Accumulation weight: " << pushConstants.accumulationWeight << std::endl;
        }
    }
    else if (frameIndex == 1)
    {
        std::cout << "[NRD] Warning: Simplified denoiser not ready - pipeline or inputs not created" << std::endl;
    }
}

void NRDWrapper::resize(uint32_t width, uint32_t height)
{
    if (m_width == width && m_height == height)
        return;
    
    std::cout << "[NRD] Resizing to " << width << "x" << height << std::endl;
    
    m_width = width;
    m_height = height;
    
    // 重建输出纹理
    destroyOutputTexture();
    createOutputTexture();
    
    // 重建历史纹理
    destroyHistoryTexture();
    createHistoryTexture();
    
    // Phase 2: 重建NRD资源
    // destroyNRDResources();
    // createNRDResources();
}

void NRDWrapper::destroy()
{
    if (!m_initialized)
        return;
    
    std::cout << "[NRD] Destroying wrapper..." << std::endl;
    
    destroyTemporalDenoisingPipeline();
    destroyHistoryTexture();
    destroyOutputTexture();
    destroyNRDResources();
    
    // 销毁NRD instance
    if (m_nrdInstance)
    {
        nrd::DestroyInstance(*m_nrdInstance);
        m_nrdInstance = nullptr;
    }
    
    m_initialized = false;
    std::cout << "[NRD] Wrapper destroyed" << std::endl;
}

void NRDWrapper::createOutputTexture()
{
    // 创建降噪输出纹理
    VkImageCreateInfo imageCI = {};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageCI.extent.width = m_width;
    imageCI.extent.height = m_height;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    if (vkCreateImage(m_device, &imageCI, nullptr, &m_outputImage) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create output image!" << std::endl;
        return;
    }
    
    // 分配内存
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device, m_outputImage, &memReqs);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_outputMemory) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to allocate output image memory!" << std::endl;
        return;
    }
    
    vkBindImageMemory(m_device, m_outputImage, m_outputMemory, 0);
    
    // 创建Image View
    VkImageViewCreateInfo viewCI = {};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = m_outputImage;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(m_device, &viewCI, nullptr, &m_outputView) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create output image view!" << std::endl;
        return;
    }
    
    std::cout << "[NRD] Output texture created" << std::endl;
}

void NRDWrapper::destroyOutputTexture()
{
    if (m_outputView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(m_device, m_outputView, nullptr);
        m_outputView = VK_NULL_HANDLE;
    }
    
    if (m_outputImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(m_device, m_outputImage, nullptr);
        m_outputImage = VK_NULL_HANDLE;
    }
    
    if (m_outputMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_outputMemory, nullptr);
        m_outputMemory = VK_NULL_HANDLE;
    }
}

void NRDWrapper::createNRDResources()
{
    // Phase 2: 创建NRD所需的Vulkan资源
    // - Descriptor pools and sets
    // - Pipeline layouts
    // - Compute pipelines (from NRD shaders)
    // - Internal textures
}

void NRDWrapper::destroyNRDResources()
{
    // Phase 2: 销毁NRD资源
    for (auto pipeline : m_pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_device, pipeline, nullptr);
    }
    m_pipelines.clear();
    
    if (m_pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    
    if (m_descriptorSet != VK_NULL_HANDLE)
    {
        // Descriptor sets are freed with pool
        m_descriptorSet = VK_NULL_HANDLE;
    }
    
    if (m_descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }
    
    if (m_descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
}

uint32_t NRDWrapper::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    
    std::cerr << "[NRD] Failed to find suitable memory type!" << std::endl;
    return 0;
}

// ==================== 简化降噪实现 ====================

void NRDWrapper::createHistoryTexture()
{
    std::cout << "[NRD] Creating history texture..." << std::endl;
    
    VkImageCreateInfo imageCI = {};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageCI.extent.width = m_width;
    imageCI.extent.height = m_height;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    if (vkCreateImage(m_device, &imageCI, nullptr, &m_historyImage) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create history image!" << std::endl;
        return;
    }
    
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device, m_historyImage, &memReqs);
    
    VkMemoryAllocateInfo memAllocInfo = {};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(m_device, &memAllocInfo, nullptr, &m_historyMemory) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to allocate history memory!" << std::endl;
        return;
    }
    
    vkBindImageMemory(m_device, m_historyImage, m_historyMemory, 0);
    
    VkImageViewCreateInfo viewCI = {};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = m_historyImage;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(m_device, &viewCI, nullptr, &m_historyView) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create history view!" << std::endl;
        return;
    }
    
    // 转换history image layout到GENERAL（compute shader需要）
    // 需要一个临时command buffer
    VkCommandPool commandPool;
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;  // 假设使用第一个queue family
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    
    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create temp command pool!" << std::endl;
        return;
    }
    
    VkCommandBuffer cmdBuffer;
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &cmdBuffer);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_historyImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    
    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
    
    vkEndCommandBuffer(cmdBuffer);
    
    // 提交并等待
    VkQueue queue;
    vkGetDeviceQueue(m_device, 0, 0, &queue);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    
    vkFreeCommandBuffers(m_device, commandPool, 1, &cmdBuffer);
    vkDestroyCommandPool(m_device, commandPool, nullptr);
    
    std::cout << "[NRD] History texture created and layout transitioned to GENERAL" << std::endl;
}

void NRDWrapper::destroyHistoryTexture()
{
    if (m_historyView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(m_device, m_historyView, nullptr);
        m_historyView = VK_NULL_HANDLE;
    }
    
    if (m_historyImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(m_device, m_historyImage, nullptr);
        m_historyImage = VK_NULL_HANDLE;
    }
    
    if (m_historyMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_historyMemory, nullptr);
        m_historyMemory = VK_NULL_HANDLE;
    }
}

void NRDWrapper::createTemporalDenoisingPipeline()
{
    std::cout << "[NRD] Creating temporal denoising pipeline..." << std::endl;
    
    // 1. 创建Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    
    for (int i = 0; i < 7; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutCI = {};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCI.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(m_device, &layoutCI, nullptr, &m_temporalDescriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create descriptor set layout!" << std::endl;
        return;
    }
    
    // 2. 创建Pipeline Layout
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 16;  // 4 * sizeof(uint32_t)
    
    VkPipelineLayoutCreateInfo pipelineLayoutCI = {};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &m_temporalDescriptorSetLayout;
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
    
    if (vkCreatePipelineLayout(m_device, &pipelineLayoutCI, nullptr, &m_temporalPipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create pipeline layout!" << std::endl;
        return;
    }
    
    // 3. 创建Descriptor Pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 7;
    
    VkDescriptorPoolCreateInfo poolCI = {};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    poolCI.maxSets = 1;
    
    if (vkCreateDescriptorPool(m_device, &poolCI, nullptr, &m_temporalDescriptorPool) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create descriptor pool!" << std::endl;
        return;
    }
    
    // 4. 分配Descriptor Set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_temporalDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_temporalDescriptorSetLayout;
    
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_temporalDescriptorSet) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to allocate descriptor set!" << std::endl;
        return;
    }
    
    // 5. 加载Shader并创建Pipeline
    // 读取编译好的SPIR-V shader
    std::string shaderPath = "../shaders/nrd_temporal_accumulation.comp.spv";
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    
    if (!file.is_open())
    {
        std::cerr << "[NRD] Warning: Could not find shader: " << shaderPath << std::endl;
        std::cerr << "[NRD] Please compile the shader using: glslc -fshader-stage=compute shaders/nrd_temporal_accumulation.comp -o " << shaderPath << std::endl;
        std::cout << "[NRD] Pipeline infrastructure created, but shader missing" << std::endl;
        return;
    }
    
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> shaderCode(fileSize);
    file.seekg(0);
    file.read(shaderCode.data(), fileSize);
    file.close();
    
    VkShaderModuleCreateInfo shaderCI = {};
    shaderCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderCI.codeSize = shaderCode.size();
    shaderCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
    
    if (vkCreateShaderModule(m_device, &shaderCI, nullptr, &m_temporalShader) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create shader module!" << std::endl;
        return;
    }
    
    // 创建Compute Pipeline
    VkPipelineShaderStageCreateInfo shaderStageCI = {};
    shaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCI.module = m_temporalShader;
    shaderStageCI.pName = "main";
    
    VkComputePipelineCreateInfo pipelineCI = {};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage = shaderStageCI;
    pipelineCI.layout = m_temporalPipelineLayout;
    
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_temporalPipeline) != VK_SUCCESS)
    {
        std::cerr << "[NRD] Failed to create compute pipeline!" << std::endl;
        return;
    }
    
    std::cout << "[NRD] Temporal denoising pipeline created successfully!" << std::endl;
}

void NRDWrapper::destroyTemporalDenoisingPipeline()
{
    if (m_temporalPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
        m_temporalPipeline = VK_NULL_HANDLE;
    }
    
    if (m_temporalShader != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(m_device, m_temporalShader, nullptr);
        m_temporalShader = VK_NULL_HANDLE;
    }
    
    if (m_temporalPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_temporalPipelineLayout, nullptr);
        m_temporalPipelineLayout = VK_NULL_HANDLE;
    }
    
    if (m_temporalDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_temporalDescriptorPool, nullptr);
        m_temporalDescriptorPool = VK_NULL_HANDLE;
        m_temporalDescriptorSet = VK_NULL_HANDLE;
    }
    
    if (m_temporalDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_temporalDescriptorSetLayout, nullptr);
        m_temporalDescriptorSetLayout = VK_NULL_HANDLE;
    }
}
