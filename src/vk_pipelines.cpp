#include "fmt/core.h"
#include <fstream>
#include <vector>
#include <vk_initializers.h>
#include <vk_pipelines.h>
#include <vulkan/vulkan_core.h>

bool vkutil::load_shader_module(const char *filePath, VkDevice device,
                                VkShaderModule *outShaderModule) {
    // Open the file with the ptr at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        fmt::println("Failed to find: {}", filePath);
        return false;
    }

    // Find what the size of the file is by looking up the location of the ptr.
    size_t fileSize = (size_t)file.tellg();

    // SpirV expects the buffer to be in uint32 type
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    // Reset the ptr to the head
    file.seekg(0);
    file.read((char *)buffer.data(), fileSize);

    // Close the file
    file.close();

    // Create the shader module
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = buffer.size() * sizeof(uint32_t), // In bytes
        .pCode = buffer.data(),                       // Actual code
    };

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
        fmt::println("Failed");
        return false;
    }
    fmt::println("Success");
    *outShaderModule = shaderModule;
    return true;
}

void vkutil::PipelineBuilder::clear() {
    this->_inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    };
    this->_rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    };
    this->_colorBlendAttachment = {};
    this->_multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    };
    this->_pipelineLayout = {};
    this->_depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    };
    this->_renderInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    };
    this->_shaderStages.clear();
}

VkPipeline vkutil::PipelineBuilder::build_pipeline(VkDevice device) {
    // make viewport state from our stored viewport and scissor
    // TODO: Support multiple viewports
    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    // For now setup dummy color blending, no transparents yet
    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &this->_colorBlendAttachment,
    };

    // Completely clear VertexInputStateCreateInfo, as we have no need for it
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    // Setup the dynamic state
    VkDynamicState state[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamicInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = &state[0],
    };

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &this->_renderInfo,
        .stageCount = (uint32_t)this->_shaderStages.size(),
        .pStages = this->_shaderStages.data(),
        .pVertexInputState = &_vertexInputInfo,
        .pInputAssemblyState = &this->_inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &this->_rasterizer,
        .pMultisampleState = &this->_multisampling,
        .pDepthStencilState = &this->_depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicInfo,
        .layout = this->_pipelineLayout,
    };

    // Now create the pipeline
    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &newPipeline) != VK_SUCCESS) {
        fmt::println("Failed to create pipeline");
        return VK_NULL_HANDLE;
    } else {
        return newPipeline;
    }
}

void vkutil::PipelineBuilder::set_shaders(VkShaderModule vertexShader,
                                          VkShaderModule fragmentShader) {
    this->_shaderStages.clear();
    this->_shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(
        VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
    this->_shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(
        VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
};
