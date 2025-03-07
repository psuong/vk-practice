#include "vk_types.h"
#include <span>
#include <vector>
#include <vk_descriptors.h>
#include <vulkan/vulkan_core.h>

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::add_bindings(uint32_t binding, VkDescriptorType type) {
    VkDescriptorSetLayoutBinding newBinding{};
    newBinding.binding = binding;
    newBinding.descriptorCount = 1;
    newBinding.descriptorType = type;

    this->bindings.push_back(newBinding);
    return *this;
}

void DescriptorLayoutBuilder::clear() { this->bindings.clear(); }

VkDescriptorSetLayout
DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages,
                               void *pNext,
                               VkDescriptorSetLayoutCreateFlags flags) {

    for (auto &b : bindings) {
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = pNext,
        .flags = flags,
        .bindingCount = (uint32_t)this->bindings.size(),
        .pBindings = this->bindings.data(),
    };

    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets,
                                    std::span<PoolSizeRatio> poolRatios) {

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type,
            .descriptorCount = uint32_t(ratio.ratio * maxSets)});
    }

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = maxSets,
        .poolSizeCount = (uint32_t)poolSizes.size(),
        .pPoolSizes = poolSizes.data()
    };

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device) {
    // Just a reset function, doesn't deallocate memory, just sets them back
    // to the initial state.
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device) {
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device,
                                              VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };

    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));
    return ds;
}
