# NRD (NVIDIA Real-time Denoiser) 集成文档

## 概述

本文档记录了在Vulkan路径追踪渲染器中集成NVIDIA NRD降噪库的完整过程，包括架构设计、关键决策和遇到的技术挑战。

---

## 目录

1. [架构设计](#架构设计)
2. [核心组件](#核心组件)
3. [关键技术决策](#关键技术决策)
4. [遇到的挑战与解决方案](#遇到的挑战与解决方案)
5. [使用说明](#使用说明)
6. [性能考虑](#性能考虑)

---

## 架构设计

### 整体流程

```
Ray Tracing Pass
    ↓ (生成NRD输入)
    ├── Radiance + HitDistance
    ├── Normal + Roughness
    ├── Motion Vector
    └── ViewZ (深度)
    ↓
NRD Denoising Pass (Compute Shader)
    ↓ (时域累积降噪)
    └── 降噪后的Radiance
    ↓
Composition Pass (最终合成)
```

### 设计原则

1. **简化实现**：使用自定义compute shader实现时域累积，而不是完整的NRD API
2. **最小侵入**：尽量不改变现有渲染流程
3. **动态切换**：支持运行时开启/关闭降噪

---

## 核心组件

### 1. NRDWrapper 类

**文件**: `src/NRDWrapper.h`, `src/NRDWrapper.cpp`

**职责**:
- 管理NRD相关的Vulkan资源
- 管理history buffer（时域累积的历史帧）
- 执行降噪compute shader

**关键成员**:

```cpp
class NRDWrapper {
private:
    // 输入纹理引用（从外部传入）
    VkImageView m_inputRadianceHitDist;    // Radiance + HitDistance
    VkImageView m_inputNormalRoughness;    // Normal + Roughness
    VkImageView m_inputMotionVector;       // Motion Vector
    VkImageView m_inputViewZ;              // View-space depth
    
    // 内部资源
    VkImage m_historyImage;                // 历史帧buffer
    VkImageView m_historyView;
    
    // Compute pipeline
    VkPipeline m_temporalPipeline;
    VkDescriptorSet m_temporalDescriptorSet;
    
    // Frame管理
    uint32_t m_internalFrameIndex;         // 内部frame counter
};
```

**为什么这样设计**:

1. **输入纹理引用而非所有权**：NRD inputs由Ray Tracing pass生成，NRDWrapper只需要读取，不需要创建和销毁
2. **内部管理history buffer**：History buffer是时域累积的核心，必须跨帧持久化，因此由NRDWrapper独立管理
3. **内部frame counter**：Command buffer录制时无法使用外部动态变化的变量，需要内部管理并在每帧录制时更新

---

### 2. NRD Input Attachments

**文件**: `src/main.cpp` (prepareOffscreenFramebuffers)

**新增的framebuffer attachments**:

```cpp
struct {
    FrameBufferAttachment radianceHitDist;   // RGBA16F: RGB=radiance, A=hitDistance
    FrameBufferAttachment normalRoughness;   // RGBA16F: RGB=normal, A=roughness
    FrameBufferAttachment motionVector;      // RG16F: 2D motion vector
    FrameBufferAttachment viewZ;             // R32F: view-space depth
} nrdInputs;
```

**关键创建参数**:

```cpp
createAttachment(
    VK_FORMAT_R16G16B16A16_SFLOAT, 
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,  // ← 关键！
    &nrdInputs.radianceHitDist, 
    width, height
);
```

**为什么需要 STORAGE_BIT**:
- Ray Tracing pass以`COLOR_ATTACHMENT`写入数据
- NRD compute shader需要以`STORAGE_IMAGE`读取数据
- 必须同时支持两种usage

---

### 3. 时域累积 Compute Shader

**文件**: `shaders/nrd_temporal_accumulation.comp`

**核心算法**:

```glsl
// 第一帧：直接使用当前帧
if (resetHistory) {
    denoisedRadiance = currentRadiance;
}
// 之后的帧：混合历史与当前
else {
    vec3 historyRadiance = imageLoad(inHistory, coord).rgb;
    float alpha = 1.0 - accumulationWeight;  // 例如0.3
    denoisedRadiance = mix(historyRadiance, currentRadiance, alpha);
}

// 输出降噪结果
imageStore(outDenoised, coord, vec4(denoisedRadiance, hitDistance));
// 更新历史（为下一帧准备）
imageStore(outHistory, coord, vec4(denoisedRadiance, hitDistance));
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `accumulationWeight` | 0.7 | 历史帧权重（70%历史 + 30%当前） |
| `resetHistory` | 0/1 | 第一帧重置历史 |

**为什么这样实现**:

1. **简化版本**：完整的NRD算法非常复杂（包括variance clipping、adaptive weighting等），简化版本易于调试和理解
2. **inHistory和outHistory是同一个buffer**：这是正确的设计！读取上一帧的历史，写入当前帧的结果作为下一帧的历史
3. **禁用motion vector重投影**：当前实现中motion vector数据可能不准确，静止相机时禁用重投影可以获得完美对齐

---

### 4. Frame Index 管理

**为什么这是最大的技术挑战**:

#### 问题：Command Buffer的录制机制

Vulkan的command buffer在**录制时**确定所有参数，之后每帧**执行相同的命令序列**。

```
录制阶段（只执行一次）:
    vkBeginCommandBuffer()
    vkCmdPushConstants(..., frameIndex=0)  // ← frameIndex在这里固定！
    vkCmdDispatch()
    vkEndCommandBuffer()

执行阶段（每帧重复）:
    vkQueueSubmit(commandBuffer)  // 执行录制好的命令，frameIndex永远是0！
```

#### 解决方案

**每帧重新录制command buffers**:

```cpp
void draw() {
    // 如果使用NRD，每帧重建command buffers
    if (useNRD) {
        buildCommandBuffers();  // ← 关键！每帧调用
    }
    
    // ... 提交command buffers
    
    // 递增frame index供下一帧使用
    if (useNRD) {
        nrdDenoiser.incrementFrameIndex();
    }
}
```

**为什么不使用全局frameNumber**:

原代码中存在一个`frameNumber`变量，但它：
- 只在相机移动或场景改变时更新
- 静止时不递增
- 导致`resetHistory`判断错误（每帧都是第0帧，每帧都重置历史）

**时间线对比**:

```
错误的实现（使用frameNumber）:
Frame 508  → resetHistory=1  ← 错误！不应该重置
Frame 649  → resetHistory=1  ← 错误！
Frame 925  → resetHistory=1  ← 错误！

正确的实现（使用nrdFrameIndex）:
Frame 0    → resetHistory=1  ← 正确！第一帧重置
Frame 1    → resetHistory=0  ← 正确！累积
Frame 2    → resetHistory=0  ← 正确！累积
Frame 3    → resetHistory=0  ← 正确！累积
```

---

## 关键技术决策

### 1. 为什么直接输出到 blur framebuffer？

```cpp
nrdDenoiser.setOutputTexture(frameBuffers.blur.color.view);
```

**原因**:
- Composition pass期望从`blur.color`读取数据
- 如果NRD输出到独立buffer，需要额外的descriptor set更新
- 直接输出避免了descriptor set的动态更新（会导致Vulkan validation错误）

**渲染流程**:

```
useNRD = false:
    Ray Tracing → Blur Pass → blur.color → Composition

useNRD = true:
    Ray Tracing → NRD (写入blur.color) → Composition
                  ↑ 跳过传统blur
```

---

### 2. 为什么只在第一次更新 Descriptor Sets？

```cpp
static bool descriptorsUpdated = false;
if (!descriptorsUpdated) {
    vkUpdateDescriptorSets(...);
    descriptorsUpdated = true;
}
```

**原因**:

Vulkan规则：**不能在command buffer pending时更新已绑定的descriptor sets**。

如果每帧都更新descriptor sets会触发validation错误：
```
VUID-vkQueueSubmit-pCommandBuffers-00070:
Descriptor set was updated while command buffer is in pending state
```

**解决方案**:
- Descriptor sets只在第一帧初始化一次
- Input/output textures的引用在整个生命周期中不变
- 只有push constants中的frameIndex每帧更新

---

### 3. Image Layout 转换策略

**NRD pass前后的layout transitions**:

```cpp
// NRD pass之前：转换到GENERAL layout（compute shader要求）
barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
// ...对所有NRD inputs + blur.color

// NRD pass之后：转换回SHADER_READ_ONLY_OPTIMAL（fragment shader使用）
barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
```

**为什么需要**:
- Fragment shader期望`SHADER_READ_ONLY_OPTIMAL`
- Compute shader要求`GENERAL`或`STORAGE_OPTIMAL`
- 必须在每个pass之间正确转换

**History buffer的特殊性**:
- History buffer始终保持`GENERAL` layout
- 不参与上述转换（它不被其他pass使用）
- 只在NRDWrapper创建时初始化一次layout

---

## 遇到的挑战与解决方案

### 挑战 1: Black Screen（黑屏）

**现象**: 开启NRD后画面全黑

**原因**:
1. 缺少`VK_IMAGE_USAGE_STORAGE_BIT`导致descriptor set创建失败
2. Image layout未正确初始化

**解决方案**:
```cpp
// 添加STORAGE_BIT
createAttachment(
    format,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,  // ← 关键
    &attachment,
    width, height
);

// 在NRD pass前后正确转换layout
vkCmdPipelineBarrier(...);  // SHADER_READ_ONLY → GENERAL
// ... NRD compute pass
vkCmdPipelineBarrier(...);  // GENERAL → SHADER_READ_ONLY
```

---

### 挑战 2: "只有第一帧降噪，之后噪声立即回来"

**现象**: 
- 开启NRD的瞬间画面变清晰
- 下一帧立即变回噪声
- 时域累积完全不工作

**调试过程**:

#### 测试 1: History Buffer持久化测试

```glsl
// 调试shader：第一帧输出红色，之后累积绿色
if (resetHistory) {
    denoisedRadiance = vec3(1.0, 0.0, 0.0);  // 红色
} else {
    float greenAccum = imageLoad(inHistory, coord).g + 0.1;
    denoisedRadiance = vec3(1.0 - greenAccum, greenAccum, 0.0);  // 逐渐变绿
}
```

**结果**: 画面从红色平滑过渡到绿色 ✅

**结论**: History buffer持久化正常，问题在别处。

#### 测试 2: Frame Index检查

**观察控制台输出**:

```
Frame 0 | resetHistory=1
Frame 0 | resetHistory=1
Frame 0 | resetHistory=1
```

**问题**: frameIndex永远是0！每帧都在重置历史！

**根本原因**: 
- Command buffers只录制一次
- Push constants在录制时固定
- 外部传入的frameIndex无法动态更新

**解决方案**:
```cpp
// 1. 每帧重新录制command buffers
void draw() {
    if (useNRD) {
        buildCommandBuffers();  // 每帧调用
    }
    // ...
    if (useNRD) {
        nrdDenoiser.incrementFrameIndex();
    }
}

// 2. NRDWrapper内部管理frameIndex
void NRDWrapper::denoise(...) {
    pushConstants.frameIndex = m_internalFrameIndex;  // 使用内部counter
    vkCmdPushConstants(...);
}
```

---

### 挑战 3: Sample Count = 1 导致效果不明显

**现象**: 即使累积正常工作，降噪效果也不明显

**原因**:
- Sample Count = 1 → 每像素只有1条光线
- 噪声极大且完全随机
- 相邻帧的噪声模式差异巨大
- 时域累积需要更多帧才能收敛

**解决方案**:

| Sample Count | 噪声水平 | 推荐用途 |
|--------------|---------|----------|
| 1 | 100% (极大) | 不推荐 |
| 4 | 50% | 测试降噪 ✅ |
| 16 | 25% | 生产环境 |
| 64+ | <10% | 参考图像 |

**结论**: **至少使用Sample Count = 4**才能看到明显的降噪效果。

---

### 挑战 4: Descriptor Set Update Timing

**Vulkan Validation错误**:
```
VUID-vkQueueSubmit-pCommandBuffers-00070:
Descriptor set 0x... was destroyed or updated between recording and 
submission of command buffer
```

**原因**: 
- 第一帧录制command buffer并绑定descriptor set
- 第二帧再次调用denoise()，尝试更新descriptor set
- 此时第一帧的command buffer可能还在GPU中pending
- Vulkan禁止这种操作

**解决方案**:
```cpp
static bool descriptorsUpdated = false;
if (!descriptorsUpdated) {
    // 只在第一次初始化descriptor sets
    vkUpdateDescriptorSets(...);
    descriptorsUpdated = true;
}
// 之后的帧重用相同的descriptor sets
```

---

## 使用说明

### 编译 NRD Shader

```bash
cd d:\PG-VXPG\VulkanPG
glslc -fshader-stage=compute shaders/nrd_temporal_accumulation.comp -o shaders/nrd_temporal_accumulation.comp.spv
```

### 运行时控制

1. **开启NRD**:
   - UI → Denoiser → 勾选 "Use NRD"
   - 会自动重置frame index

2. **调整降噪强度**:
   ```cpp
   // 在 NRDWrapper.cpp 中修改
   pushConstants.accumulationWeight = 0.7f;  // 0.7 = 70%历史 + 30%当前
   ```
   - 值越大，累积越慢，但更稳定
   - 值越小，响应越快，但可能有轻微闪烁

3. **最佳测试条件**:
   - Sample Count = 4 或更高
   - 完全静止相机（不要移动鼠标）
   - 观察30秒，噪声会逐渐减少

### 预期效果

**时间轴（Sample Count = 4，静止相机）**:

| 时间 | 噪声水平 | 视觉效果 |
|------|---------|----------|
| 0秒 (第1帧) | 100% | 明显颗粒感 |
| 3秒 | 60% | 颗粒减少 |
| 10秒 | 30% | 较为平滑 |
| 30秒 | <15% | 接近无噪声 |

---

## 性能考虑

### 当前实现的开销

1. **每帧重建 Command Buffers** (~0.5-1ms)
   - 为了动态更新frameIndex
   - 可以通过descriptor indexing优化

2. **Compute Shader Dispatch** (~0.5ms)
   - 1280x720 @ 16x16 workgroup
   - 主要时间花在texture读写

3. **额外内存**:
   - History buffer: width × height × 4 × sizeof(float16) ≈ 7MB @ 1280x720

### 优化建议

#### 1. 使用 Descriptor Indexing（未实现）

```cpp
// 避免每帧重建command buffers
layout(set = 0, binding = 0) buffer FrameIndexBuffer {
    uint frameIndex;
} frameIndexBuffer;

// GPU直接读取动态更新的buffer
uint currentFrame = frameIndexBuffer.frameIndex;
```

#### 2. 恢复 Motion Vector 重投影（未实现）

当前禁用了motion vector，只在静止相机时有效。正确实现motion vector后可以支持动态场景。

#### 3. 添加 Variance Clipping（已删除，可恢复）

防止ghosting（拖影）问题：

```glsl
// 计算当前帧邻域的方差
vec3 m1 = ..., m2 = ...;
vec3 sigma = sqrt(m2 - m1 * m1);

// 裁剪历史到合理范围
vec3 minVal = m1 - sigma * 1.5;
vec3 maxVal = m1 + sigma * 1.5;
historyRadiance = clamp(historyRadiance, minVal, maxVal);
```

---

## 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/NRDWrapper.h` | NRD包装类头文件 |
| `src/NRDWrapper.cpp` | NRD包装类实现 |
| `shaders/nrd_temporal_accumulation.comp` | 时域累积compute shader |
| `shaders/nrd_temporal_accumulation.comp.spv` | 编译后的shader |

### 修改文件

| 文件 | 主要修改 |
|------|----------|
| `src/main.cpp` | • 添加NRD inputs attachments<br>• 添加NRD pass到渲染流程<br>• 每帧重建command buffers<br>• UI控制 |
| `shaders/pathtracing.frag` | • 输出NRD所需的数据<br>• Radiance + HitDistance<br>• Normal + Roughness<br>• Motion Vector<br>• ViewZ |

---

## 调试技巧

### 1. 验证 History Buffer 持久化

在shader中添加：

```glsl
// 强制第一帧输出红色
if (resetHistory) {
    denoisedRadiance = vec3(1.0, 0.0, 0.0);
}
// 后续帧逐渐变绿
else {
    float green = imageLoad(inHistory, coord).g + 0.1;
    denoisedRadiance = vec3(1.0 - green, green, 0.0);
}
```

**预期**: 画面从红色平滑过渡到绿色

### 2. 检查 Frame Index

添加控制台输出：

```cpp
std::cout << "[NRD] Frame " << m_internalFrameIndex 
          << " | resetHistory=" << pushConstants.resetHistory << std::endl;
```

**预期**: 
```
Frame 0 | resetHistory=1
Frame 1 | resetHistory=0
Frame 2 | resetHistory=0
...
```

### 3. 验证 Descriptor Sets

```cpp
if (frameIndex <= 1) {
    std::cout << "[NRD] Output view: " 
              << (m_outputView != VK_NULL_HANDLE ? "Valid" : "Invalid") << std::endl;
    std::cout << "[NRD] History view: " 
              << (m_historyView != VK_NULL_HANDLE ? "Valid" : "Invalid") << std::endl;
}
```

---

## 已知限制

1. **静态场景优化**: 当前禁用motion vector，只在静态相机时效果最佳
2. **每帧重建开销**: 需要每帧重建command buffers，有一定CPU开销
3. **简化算法**: 未实现完整NRD算法（如SVGF、REBLUR等）
4. **Sample Count依赖**: 需要较高的sample count（≥4）才能看到明显效果

---

## 未来改进

1. [ ] 实现正确的motion vector计算和重投影
2. [ ] 使用descriptor indexing避免每帧重建command buffers
3. [ ] 添加variance clipping防止ghosting
4. [ ] 支持完整的NRD API（REBLUR/RELAX等方法）
5. [ ] 添加adaptive accumulation weight（根据roughness调整）
6. [ ] 支持多种降噪模式切换

---

## 参考资料

- [NVIDIA NRD Library](https://github.com/NVIDIAGameWorks/RayTracingDenoiser)
- [Vulkan Specification](https://www.khronos.org/registry/vulkan/specs/1.3/html/)
- [Temporal Anti-Aliasing (TAA) 原理](https://developer.nvidia.com/sites/default/files/akamai/gameworks/blog/TAA.pdf)

---

## 总结

NRD集成的核心挑战是**在Vulkan的command buffer机制下实现动态参数更新**。通过每帧重建command buffers和内部管理frame index，我们成功实现了基础的时域累积降噪。

**关键要点**:
1. ✅ History buffer跨帧持久化
2. ✅ Frame index正确递增（0, 1, 2, ...）
3. ✅ Descriptor sets只初始化一次
4. ✅ Image layouts正确转换
5. ✅ 需要合理的sample count（≥4）

**最终效果**: 在静止相机、Sample Count ≥ 4的条件下，噪声在30秒内可减少80%以上。
