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
