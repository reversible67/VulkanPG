#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>

// Forward declarations
namespace nrd { struct Instance; }

// NRD Wrapper for VulkanPG
class NRDWrapper
{
public:
    NRDWrapper() = default;
    ~NRDWrapper() { destroy(); }

    // 初始化（Phase 1：占位实现，Phase 2：完整实现）
    bool initialize(VkDevice device, VkPhysicalDevice physDevice, 
                   uint32_t width, uint32_t height);
    
    // 设置输入纹理
    void setInputTextures(
        VkImageView radianceHitDist,      // RGBA16F: RGB=radiance, A=hitDistance
        VkImageView normalRoughness,      // RGBA16F: RGB=normal01, A=roughness
        VkImageView viewZ,                // R32F: view space Z
        VkImageView motionVector          // RG16F: screen space motion
    );
    
    // 设置输出纹理（让NRD直接写入到指定位置，如blur的framebuffer）
    void setOutputTexture(VkImageView outputView);
    
    // 执行降噪
    void denoise(VkCommandBuffer cmd, uint32_t frameIndex,
                const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                const glm::mat4& prevViewMatrix, const glm::mat4& prevProjMatrix);
    
    // 重置内部frame counter
    void resetFrameIndex() { m_internalFrameIndex = 0; }
    
    // 递增内部frame counter
    void incrementFrameIndex() { m_internalFrameIndex++; }
    
    // 获取降噪后的输出
    VkImageView getOutputTexture() const { return m_outputView; }
    
    // 窗口大小改变
    void resize(uint32_t width, uint32_t height);
    
    // 清理资源
    void destroy();
    
    // 检查是否已初始化
    bool isInitialized() const { return m_initialized; }

private:
    // Vulkan基础
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_initialized = false;
    
    // NRD实例
    nrd::Instance* m_nrdInstance = nullptr;
    
    // 输入纹理
    VkImageView m_inputRadianceHitDist = VK_NULL_HANDLE;
    VkImageView m_inputNormalRoughness = VK_NULL_HANDLE;
    VkImageView m_inputViewZ = VK_NULL_HANDLE;
    VkImageView m_inputMotionVector = VK_NULL_HANDLE;
    
    // 输出纹理
    VkImage m_outputImage = VK_NULL_HANDLE;
    VkDeviceMemory m_outputMemory = VK_NULL_HANDLE;
    VkImageView m_outputView = VK_NULL_HANDLE;
    
    // 简化降噪：历史buffer
    VkImage m_historyImage = VK_NULL_HANDLE;
    VkDeviceMemory m_historyMemory = VK_NULL_HANDLE;
    VkImageView m_historyView = VK_NULL_HANDLE;
    
    // 内部frame counter（command buffer录制时无法动态更新外部传入的frameIndex）
    uint32_t m_internalFrameIndex = 0;
    
    // 简化降噪：compute pipeline
    VkShaderModule m_temporalShader = VK_NULL_HANDLE;
    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_temporalPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_temporalDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_temporalDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_temporalDescriptorSet = VK_NULL_HANDLE;
    
    // 完整NRD资源（Phase 2会使用）
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    std::vector<VkPipeline> m_pipelines;
    
    // 辅助函数
    void createOutputTexture();
    void destroyOutputTexture();
    void createHistoryTexture();
    void destroyHistoryTexture();
    void createTemporalDenoisingPipeline();
    void destroyTemporalDenoisingPipeline();
    void createNRDResources();      // Phase 2实现
    void destroyNRDResources();     // Phase 2实现
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};
