//> includes
#include "SDL_video.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/gtx/transform.hpp"
#include "vk_loader.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
#include <vulkan/vulkan_core.h>
#define VMA_IMPLEMENTATION
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include "utils.h"
#include "vk_engine.h"
#include "vk_mem_alloc.h"
#include "vk_pipelines.h"
#include "vk_types.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"
#include "vk_images.h"

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() {
    return *loadedEngine;
}

void VulkanEngine::init() {
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, window_flags);

    this->init_vulkan();
    this->init_swapchain();
    this->init_commands();
    this->init_sync_structures();
    this->init_descriptors();

    this->init_pipelines();
    this->init_imgui();
    this->init_default_data();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;

    // Use the debugger
    auto inst = builder.set_app_name("Vk Renderer")
                    .request_validation_layers(true)
                    .use_default_debug_messenger() // TODO: Write my own debug callback
                    // to print out the line #
                    .require_api_version(1, 3, 0)
                    .build();

    vkb::Instance vkb_inst = inst.value();
    this->_instance = vkb_inst.instance;
    this->_debugMessenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(this->_window, this->_instance, &_surface);

    VkPhysicalDeviceFeatures features1_0{.shaderInt64 = VK_TRUE};

    VkPhysicalDeviceVulkan11Features features1_1{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                                 .shaderDrawParameters = true};

    VkPhysicalDeviceVulkan12Features features1_2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .descriptorIndexing = true,
        .bufferDeviceAddress = true,
    };

    VkPhysicalDeviceVulkan13Features features1_3{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = true,
        .dynamicRendering = true,
    };

    vkb::PhysicalDeviceSelector selector{vkb_inst};

    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 3)
                                             .set_required_features(features1_0)
                                             .set_required_features_11(features1_1)
                                             .set_required_features_12(features1_2)
                                             .set_required_features_13(features1_3)
                                             .set_surface(this->_surface)
                                             .select()
                                             .value();

    // Create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of the vulkan application
    this->_device = vkbDevice.device;
    this->_chosenGPU = physicalDevice.physical_device;

    this->_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    this->_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = this->_chosenGPU;
    allocatorInfo.device = this->_device;
    allocatorInfo.instance = this->_instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &this->_allocator);

    this->_mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(this->_allocator); });
}

void VulkanEngine::init_swapchain() {
    this->create_swapchain(this->_windowExtent.width, this->_windowExtent.height);
    VkExtent3D drawImageExtent = {this->_windowExtent.width, this->_windowExtent.height, 1};

    // 64 bits per pixel
    this->_drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    this->_drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimgInfo =
        vkinit::image_create_info(this->_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // Allocate the image on gpu local memory
    VmaAllocationCreateInfo rimgAllocInfo = {};
    rimgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimgAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vmaCreateImage(this->_allocator, &rimgInfo, &rimgAllocInfo, &_drawImage.image, &_drawImage.allocation,
                   VK_NULL_HANDLE);

    // Allocate and create the image
    VkImageViewCreateInfo rViewInfo =
        vkinit::imageview_create_info(this->_drawImage.imageFormat, this->_drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(this->_device, &rViewInfo, VK_NULL_HANDLE, &this->_drawImage.imageView));

    this->_depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    this->_depthImage.imageExtent = drawImageExtent;

    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info =
        vkinit::image_create_info(this->_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // allocate and create the image
    vmaCreateImage(this->_allocator, &dimg_info, &rimgAllocInfo, &this->_depthImage.image,
                   &this->_depthImage.allocation, nullptr);

    // build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
        this->_depthImage.imageFormat, this->_depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    // Add to the deletion queue because we've allocated memory on the gpu
    this->_mainDeletionQueue.push_function([=, this]() {
        vkDestroyImageView(this->_device, this->_drawImage.imageView, VK_NULL_HANDLE);
        vmaDestroyImage(this->_allocator, this->_drawImage.image, this->_drawImage.allocation);

        vkDestroyImageView(this->_device, this->_depthImage.imageView, VK_NULL_HANDLE);
        vmaDestroyImage(this->_allocator, this->_depthImage.image, this->_depthImage.allocation);
    });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{this->_chosenGPU, this->_device, this->_surface};
    // Because we need to draw imgui which requires the alpha channel, we
    // change from
    // VK_FORMAT_B8G8R8_UNORM
    // VK_FORMAT_B8G8R8A8_UNORM
    this->_swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM; // 32 bit format

    vkb::Swapchain vkbSwapchain =
        swapchainBuilder
            .set_desired_format(VkSurfaceFormatKHR{.format = this->_swapchainImageFormat,
                                                   .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    this->_swapchainExtent = vkbSwapchain.extent;
    this->_swapchain = vkbSwapchain.swapchain;
    this->_swapchainImages = vkbSwapchain.get_images().value();
    this->_swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain() {
    vkDestroySwapchainKHR(this->_device, this->_swapchain, nullptr);

    for (int i = 0; i < this->_swapchainImageViews.size(); i++) {
        vkDestroyImageView(this->_device, this->_swapchainImageViews[i], nullptr);
    }
}

void VulkanEngine::init_commands() {
    VkCommandPoolCreateInfo commandPoolInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                               .pNext = nullptr,
                                               .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                               // We say that the command buffer is compatible with the graphics queue
                                               // family
                                               .queueFamilyIndex = this->_graphicsQueueFamily};

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(this->_device, &commandPoolInfo, nullptr, &this->_frames[i]._commandPool));

        // Allocate the default cmd buffer to use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo =
            vkinit::command_buffer_allocate_info(this->_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(this->_device, &cmdAllocInfo, &this->_frames[i]._mainCommandBuffer));
    }

    // > imgui
    // Allocate command buffer for immediate submissions
    VK_CHECK(vkCreateCommandPool(this->_device, &commandPoolInfo, nullptr, &this->_immCommandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(this->_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(this->_device, &cmdAllocInfo, &this->_immCommandBuffer));

    this->_mainDeletionQueue.push_function(
        [=, this]() { vkDestroyCommandPool(this->_device, this->_immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures() {
    // First fence controls when the gpu finished rendering the frame
    // 2 semaphore to synchronize rendering with swapchain
    // The fence signals so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        FrameData& frameData = this->_frames[i];
        VK_CHECK(vkCreateFence(this->_device, &fenceCreateInfo, nullptr, &frameData._renderFence));

        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr, &frameData._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr, &frameData._renderSemaphore));
    }

    // > imgui
    // Initialize imgui's fence
    VK_CHECK(vkCreateFence(this->_device, &fenceCreateInfo, nullptr, &this->_immFence));
    this->_mainDeletionQueue.push_function([=, this]() { vkDestroyFence(this->_device, this->_immFence, nullptr); });
}

void VulkanEngine::init_descriptors() {
    // Create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};

    this->globalDescriptorAllocator.init_pool(this->_device, 10, sizes);

    // Make the descriptor set layout for our compute draw, bind to the first
    // layer
    DescriptorLayoutBuilder builder;
    builder = builder.add_bindings(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    this->_drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    // Allocate a descriptor for our draw image
    this->_drawImageDescriptors =
        this->globalDescriptorAllocator.allocate(this->_device, this->_drawImageDescriptorLayout);

    VkDescriptorImageInfo imgInfo{
        .imageView = this->_drawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    VkWriteDescriptorSet drawImageWrite = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .pNext = nullptr,
                                           .dstSet = this->_drawImageDescriptors,
                                           .dstBinding = 0,
                                           .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .pImageInfo = &imgInfo};

    vkUpdateDescriptorSets(this->_device, 1, &drawImageWrite, 0, nullptr);

    // Clean up the descriptor allocator and new layout
    this->_mainDeletionQueue.push_function([&]() {
        this->globalDescriptorAllocator.destroy_pool(this->_device);
        vkDestroyDescriptorSetLayout(this->_device, this->_drawImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::init_pipelines() {
    this->init_background_pipelines();
    this->init_triangle_pipeline();
    this->init_mesh_pipeline();
}

void VulkanEngine::init_triangle_pipeline() {
    VkShaderModule triangleFragShader;
    char buffer[MAX_PATH];
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\colored_triangle_frag.spv"),
                                    this->_device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module\n");
    } else {
        fmt::print("Triangle fragment shader successfully loaded\n");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\colored_triangle_vert.spv"),
                                    this->_device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader module\n");
    } else {
        fmt::print("Triangle vertex shader successfully loaded\n");
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    vkutil::PipelineBuilder pipelineBuilder;
    pipelineBuilder._pipelineLayout = this->_trianglePipelineLayout;
    this->_trianglePipeline = pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader)
                                  .set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                                  .set_polygon_mode(VK_POLYGON_MODE_FILL)
                                  .set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
                                  .set_multisampling_none()
                                  .disable_blending()
                                  .disable_depthtest()
                                  .set_color_attachment_format(this->_drawImage.imageFormat)
                                  .set_depth_format(VK_FORMAT_D32_SFLOAT)
                                  .enable_blending_additive()
                                  .build_pipeline(this->_device, "triangle_pipeline");

    vkDestroyShaderModule(this->_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(this->_device, triangleVertexShader, nullptr);

    this->_mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(this->_device, this->_trianglePipelineLayout, nullptr);
        if (this->_trianglePipeline != VK_NULL_HANDLE) {
            fmt::println("Destroying triangle pipeline");
            vkDestroyPipeline(this->_device, this->_trianglePipeline, nullptr);
        }
    });
}

void VulkanEngine::init_mesh_pipeline() {
    VkShaderModule triangleFragShader;
    char buffer[MAX_PATH];
    if (!vkutil::load_shader_module(
            utils::get_relative_path(buffer, MAX_PATH, "shaders\\colored_triangle_mesh_frag.spv"), _device,
            &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module");
    } else {
        fmt::print("Triangle fragment shader succesfully loaded");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module(
            utils::get_relative_path(buffer, MAX_PATH, "shaders\\colored_triangle_mesh_vert.spv"), this->_device,
            &triangleVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module.");
    } else {
        fmt::println("Triangle vertex shader successfully loaded.");
    }

    VkPushConstantRange bufferRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(GPUDrawPushConstants)};

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    vkutil::PipelineBuilder pipelineBuilder;
    pipelineBuilder._pipelineLayout = this->_meshPipelineLayout;

    this->_meshPipeline = pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader)
                              .set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                              .set_polygon_mode(VK_POLYGON_MODE_FILL)
                              .set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
                              .set_multisampling_none()
                              .disable_blending()
                              .enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL)
                              .set_color_attachment_format(this->_drawImage.imageFormat)
                              .set_depth_format(this->_depthImage.imageFormat)
                              .build_pipeline(this->_device, "mesh_pipeline");

    // Destroy the shaders
    vkDestroyShaderModule(this->_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(this->_device, triangleVertexShader, nullptr);

    this->_mainDeletionQueue.push_function([&, this]() {
        vkDestroyPipelineLayout(this->_device, this->_meshPipelineLayout, nullptr);
        vkDestroyPipeline(this->_device, this->_meshPipeline, nullptr);
    });
}

void VulkanEngine::init_background_pipelines() {
    // TODO: Seems duplicate, need to figure out what im doing with the compute
    // layouts...
    VkPushConstantRange pushConstants{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputePushConstants),
    };

    VkPipelineLayoutCreateInfo computeLayout{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &this->_drawImageDescriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstants,
    };
    VK_CHECK(vkCreatePipelineLayout(this->_device, &computeLayout, nullptr, &this->_gradientPipelineLayout));

    // TODO: Load the sky shader module
    VkShaderModule gradientShader;
    char buffer[MAX_PATH];
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\gradient_color.comp.spv"),
                                    this->_device, &gradientShader)) {
        fmt::println("[ERROR] Cannot build the gradient compute shader");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\sky.comp.spv"), this->_device,
                                    &skyShader)) {
        fmt::println("[ERROR] Cannot build the sky compute shader");
    }

    VkPipelineShaderStageCreateInfo stageInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .module = gradientShader,
                                              .pName = "main"};

    VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = stageInfo,
        .layout = this->_gradientPipelineLayout,
    };

    ComputeEffect gradient{
        .name = "gradient",
        .layout = this->_gradientPipelineLayout,
        .data =
            {
                // Default colors
                .data1 = glm::vec4(1, 0, 0, 1),
                .data2 = glm::vec4(0, 0, 1, 1),
            },
    };

    if (vkCreateComputePipelines(this->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr,
                                 &gradient.pipeline) == VK_SUCCESS) {
        utils::set_pipeline_debug_name(this->_device, (uint64_t)gradient.pipeline, VK_OBJECT_TYPE_PIPELINE, "gradient");
    }

    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky{
        .name = "sky", .layout = this->_gradientPipelineLayout, .data = {.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97)}};

    if (vkCreateComputePipelines(this->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr,
                                 &sky.pipeline) == VK_SUCCESS) {
        fmt::println("Sky pipeline created");
        utils::set_pipeline_debug_name(this->_device, (uint64_t)sky.pipeline, VK_OBJECT_TYPE_PIPELINE, "sky");
    }

    this->backgroundEffects.push_back(gradient);
    this->backgroundEffects.push_back(sky);

    // Clean up the compute pipeline
    vkDestroyShaderModule(this->_device, gradientShader, nullptr);
    vkDestroyShaderModule(this->_device, skyShader, nullptr);
    this->_mainDeletionQueue.push_function([=, this]() {
        vkDestroyPipelineLayout(this->_device, this->_gradientPipelineLayout, nullptr);
        if (sky.pipeline != VK_NULL_HANDLE) {
            fmt::println("Destroying sky pipeline");
            vkDestroyPipeline(this->_device, sky.pipeline, nullptr);
        }

        if (gradient.pipeline != VK_NULL_HANDLE) {
            fmt::println("Destroying gradient pipeline");
            vkDestroyPipeline(this->_device, gradient.pipeline, nullptr);
        }
    });
}

void VulkanEngine::init_default_data() {
    std::array<Vertex, 4> rect_vertices;
    rect_vertices[0].position = {0.5, -0.5, 0};
    rect_vertices[1].position = {0.5, 0.5, 0};
    rect_vertices[2].position = {-0.5, -0.5, 0};
    rect_vertices[3].position = {-0.5, 0.5, 0};

    rect_vertices[0].color = {0, 0, 0, 1};
    rect_vertices[1].color = {0.5, 0.5, 0.5, 1};
    rect_vertices[2].color = {1, 0, 0, 1};
    rect_vertices[3].color = {0, 1, 0, 1};

    std::array<uint32_t, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = this->upload_mesh(rect_indices, rect_vertices);

    this->_mainDeletionQueue.push_function([&]() {
        this->destroy_buffer(rectangle.indexBuffer);
        this->destroy_buffer(rectangle.vertexBuffer);
    });

    char buffer[MAX_PATH];

    const char* basic_mesh_path = utils::get_relative_path(buffer, MAX_PATH, "assets\\basicmesh.glb");
    this->testMeshes = loadGltfMeshes(this, buffer).value();
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) {
    VK_CHECK(vkResetFences(this->_device, 1, &this->_immFence));
    VK_CHECK(vkResetCommandBuffer(this->_immCommandBuffer, 0));

    VkCommandBuffer cmd = this->_immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    // Submit command buffer to the queue and immediate execute it
    // _renderFence is not blocked until the graphics cmds finish executing with
    // the imm buffer
    VK_CHECK(vkQueueSubmit2(this->_graphicsQueue, 1, &submit, this->_immFence));

    VK_CHECK(vkWaitForFences(this->_device, 1, &this->_immFence, true, 99999999));
}

void VulkanEngine::init_imgui() {
    // Create the descriptor pool
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo poolInfo = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                           .maxSets = 1000,
                                           .poolSizeCount = (uint32_t)std::size(poolSizes),
                                           .pPoolSizes = poolSizes};

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(this->_device, &poolInfo, nullptr, &imguiPool));

    // Initialize the imgui lib
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForVulkan(this->_window);

    ImGui_ImplVulkan_InitInfo initInfo = {.Instance = this->_instance,
                                          .PhysicalDevice = this->_chosenGPU,
                                          .Device = this->_device,
                                          .Queue = this->_graphicsQueue,
                                          .DescriptorPool = imguiPool,
                                          .MinImageCount = 3,
                                          .ImageCount = 3,
                                          .UseDynamicRendering = true};

    // Use dynamic rendering params for imgui
    initInfo.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &this->_swapchainImageFormat,
    };

    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);
    ImGui_ImplVulkan_CreateFontsTexture();

    this->_mainDeletionQueue.push_function([=, this]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(this->_device, imguiPool, nullptr);
    });
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(this->_device);

        // Free the command pool
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            FrameData& frameData = this->_frames[i];
            vkDestroyCommandPool(this->_device, frameData._commandPool, nullptr);

            // Destroy the sync objects
            vkDestroyFence(this->_device, frameData._renderFence, nullptr);
            vkDestroySemaphore(this->_device, frameData._renderSemaphore, nullptr);
            vkDestroySemaphore(this->_device, frameData._swapchainSemaphore, nullptr);

            frameData._deletionQueue.flush();
        }

        // Cleanup all of the loaded meshes
        for (auto& mesh : this->testMeshes) {
            this->destroy_buffer(mesh->meshBuffers.indexBuffer);
            this->destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }

        this->_mainDeletionQueue.flush();

        // Perform all of the clean up operations
        destroy_swapchain();

        vkDestroySurfaceKHR(this->_instance, this->_surface, nullptr);
        fmt::println("Destroying device");
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(this->_instance, this->_debugMessenger);
        vkDestroyInstance(this->_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw() {
    FrameData& frameData = get_current_frame();
    VkDevice device = this->_device;

    // Wait until the GPU finished rendering the last frame, with a timeout of 1
    // second
    VK_CHECK(vkWaitForFences(device, 1, &frameData._renderFence, true, 1000000000));

    // Flush the deletion queue
    frameData._deletionQueue.flush();

    VK_CHECK(vkResetFences(device, 1, &frameData._renderFence));

    uint32_t swapchainImageIndex;
    // When acquiring the image from the swapchain, we request an available one.
    // We have a timeout to wait until the next image is available if there is
    // none available.
    VkResult err = vkAcquireNextImageKHR(device, this->_swapchain, 1000000000, frameData._swapchainSemaphore,
                                         VK_NULL_HANDLE, &swapchainImageIndex);

    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    VkCommandBuffer cmd = frameData._mainCommandBuffer;
    // Ensure that our cmd buffer is resetted
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin recording, we use the command buffer exactly once
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    this->_drawExtent.width =
        min(this->_swapchainExtent.width, this->_drawImage.imageExtent.width) * this->renderScale;
    this->_drawExtent.height =
        min(this->_swapchainExtent.height, this->_drawImage.imageExtent.height) * this->renderScale;

    // Begin recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition the main draw image into a general layout to write to it
    // Then overwrite it all, b/c we dont care about what the older layout was
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    this->draw_background(cmd);

    // For start, transition the swapchain image to a drawable layout, then
    // clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, this->_depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    this->draw_geometry(cmd);

    // Change the swapchain back to a format that works for displays
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // > Begin drawing imgui
    // Execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, this->_drawImage.image, this->_swapchainImages[swapchainImageIndex],
                                this->_drawExtent, this->_swapchainExtent);

    // Set the swapchain image layout to present so we can show it on the
    // screen.
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // draw imgui into the swapchain
    this->draw_imgui(cmd, this->_swapchainImageViews[swapchainImageIndex]);

    // set the swapchain image layout to present to actually be able to show it
    // on screen
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));
    // < End drawing imgui

    // Prepare the submission to the queue
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    // Need to wait until the presentSemaphore (this will tell us that the
    // swapchain is ready)
    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frameData._swapchainSemaphore);

    // The render semaphore signals that the rendering finished
    VkSemaphoreSubmitInfo signalInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR, frameData._renderSemaphore);

    // Submit the command buffer to the queue and execute it
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // Block the render fence until the graphics commands finish executing
    VK_CHECK(vkQueueSubmit2(this->_graphicsQueue, 1, &submit, frameData._renderFence));

    // Present the image

    VkPresentInfoKHR presentInfo = vkinit::present_info();

    // we have a single swapchain
    presentInfo.pSwapchains = &this->_swapchain;
    presentInfo.swapchainCount = 1;

    // we have a wait semaphore
    presentInfo.pWaitSemaphores = &frameData._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(this->_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }

    this->_frameNumber++;
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) {
    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(this->_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {
    VkClearColorValue clearValue;
    float flash = std::abs(std::sin(this->_frameNumber / 120.f));
    clearValue = {{0.0f, 0.0f, flash, 1.0f}};

    VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

    ComputeEffect& effect = this->backgroundEffects[this->currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->_gradientPipelineLayout, 0, 1,
                            &this->_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, this->_gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants),
                       &effect.data);

    // Execute the compute pipeline dispatch. We are using 16x16
    // workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(this->_drawExtent.width / 16.0), std::ceil(this->_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {
    // begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(this->_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo depthAttachment =
        vkinit::depth_attachment_info(this->_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);

    // set dynamic viewport and scissor
    VkViewport viewport = {.x = 0,
                           .y = 0,
                           .width = (float)this->_drawExtent.width,
                           .height = (float)this->_drawExtent.height,
                           .minDepth = 0.0f,
                           .maxDepth = 1.0f};

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {.offset = VkOffset2D{.x = 0, .y = 0},
                        .extent = VkExtent2D{.width = this->_drawExtent.width, .height = this->_drawExtent.height}};

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // launch a draw command to draw 3 vertices
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, this->_meshPipeline);

    GPUDrawPushConstants push_constants{.worldMatrix = glm::mat4{1.f}, .vertexBuffer = rectangle.vertexBufferAddress};

    vkCmdPushConstants(cmd, this->_meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants),
                       &push_constants);
    vkCmdBindIndexBuffer(cmd, this->rectangle.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    // Draw the monkey head
    glm::mat4 view = glm::translate(glm::vec3(0, 0, -5));
    glm::mat4 projection = glm::perspective(
        glm::radians(70.0f), (float)this->_drawExtent.width / (float)this->_drawExtent.height, 10000.f, 0.1f);

    projection[1][1] *= -1;
    push_constants.worldMatrix = projection * view;

    push_constants.vertexBuffer = this->testMeshes[2]->meshBuffers.vertexBufferAddress;

    vkCmdPushConstants(cmd, this->_meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants),
                       &push_constants);
    vkCmdBindIndexBuffer(cmd, this->testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, this->testMeshes[2]->surfaces[0].count, 1, this->testMeshes[2]->surfaces[0].start_index, 0,
                     0);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }

            // Send sdl events to imgui
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (this->resize_requested) {
            this->resize_swapchain();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // ImGui::ShowDemoWindow();
        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &this->renderScale, 0.3, 1.0f);
            ComputeEffect& selected = this->backgroundEffects[this->currentBackgroundEffect];
            ImGui::Text("Selected effect: ", selected.name);
            ImGui::SliderInt("Effect Index", &this->currentBackgroundEffect, 0, this->backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);
            ImGui::InputFloat4("data3", (float*)&selected.data.data3);
            ImGui::InputFloat4("data4", (float*)&selected.data.data4);
        }
        ImGui::End();

        ImGui::Render();

        draw();
    }
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                                            const char* name) {
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = nullptr, .size = allocSize, .usage = usage};

    VmaAllocationCreateInfo vmaAllocInfo = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = memoryUsage,
    };

    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(this->_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation,
                             &newBuffer.info));

    VkDebugUtilsObjectNameInfoEXT nameInfo{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                           .objectType = VK_OBJECT_TYPE_BUFFER,
                                           .objectHandle = (uint64_t)newBuffer.buffer,
                                           .pObjectName = name};
    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer) {
    vmaDestroyBuffer(this->_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer = create_buffer(vertexBufferSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            VMA_MEMORY_USAGE_GPU_ONLY, "vertex_buffer");
    VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                .buffer = newSurface.vertexBuffer.buffer};
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(this->_device, &deviceAddressInfo);

    newSurface.indexBuffer =
        create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY, "index_buffer");

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VMA_MEMORY_USAGE_CPU_ONLY, "staging_buffer");

    void* data = staging.allocation->GetMappedData();
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertexBufferSize,
        };
        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{.srcOffset = vertexBufferSize, .dstOffset = 0, .size = indexBufferSize};
        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    this->destroy_buffer(staging);
    return newSurface;
}

void VulkanEngine::resize_swapchain() {
    vkDeviceWaitIdle(this->_device);
    this->destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(this->_window, &w, &h);
    this->_windowExtent.width = w;
    this->_windowExtent.height = h;

    this->create_swapchain(this->_windowExtent.width, this->_windowExtent.height);
    this->resize_requested = false;
}
