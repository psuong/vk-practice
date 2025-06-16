#pragma once
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

namespace vkutil {

bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);

class PipelineBuilder {
  public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;

    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipelineRenderingCreateInfo _renderInfo;
    VkFormat _colorAttachmentFormat;

    PipelineBuilder() {
        this->clear();
    };

    void clear();

    VkPipeline build_pipeline(VkDevice device, const char* pipelineName);
    PipelineBuilder& set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
    PipelineBuilder& set_input_topology(VkPrimitiveTopology topology);
    PipelineBuilder& set_polygon_mode(VkPolygonMode mode);
    PipelineBuilder& set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    PipelineBuilder& set_multisampling_none();
    PipelineBuilder& disable_blending();
    PipelineBuilder& set_color_attachment_format(VkFormat format);
    PipelineBuilder& set_depth_format(VkFormat format);
    PipelineBuilder& disable_depthtest();
};
}; // namespace vkutil
