//> includes
#include "SDL_video.h"
#include "fmt/core.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/packing.hpp"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
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
    this->init_default_data();
    this->init_renderables();
    this->init_imgui();

    // everything went fine
    this->_isInitialized = true;
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

    // Initialize imgui's fence
    VK_CHECK(vkCreateFence(this->_device, &fenceCreateInfo, nullptr, &this->_immFence));
    this->_mainDeletionQueue.push_function([=, this]() { vkDestroyFence(this->_device, this->_immFence, nullptr); });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        FrameData& frameData = this->_frames[i];
        VK_CHECK(vkCreateFence(this->_device, &fenceCreateInfo, nullptr, &frameData._renderFence));

        VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();
        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr, &frameData._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr, &frameData._renderSemaphore));

        this->_mainDeletionQueue.push_function([=, this]() {
            vkDestroyCommandPool(this->_device, this->_frames[i]._commandPool, nullptr);
            vkDestroyFence(this->_device, this->_frames[i]._renderFence, nullptr);
            vkDestroySemaphore(this->_device, this->_frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(this->_device, this->_frames[i]._swapchainSemaphore, nullptr);
        });
    }
}

void VulkanEngine::init_descriptors() {
    // Create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                                                     {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};

    this->globalDescriptorAllocator.init(this->_device, 10, sizes);
    {
        // Make the descriptor set layout for our compute draw, bind to the first layer
        DescriptorLayoutBuilder builder;
        this->_drawImageDescriptorLayout =
            builder.add_bindings(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).build(this->_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        this->_singleImageDescriptorLayout = builder.add_bindings(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                                                 .build(this->_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        this->_gpuSceneDataDescriptorLayout =
            builder.add_bindings(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .build(this->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    // Allocate a descriptor for our draw image
    this->_drawImageDescriptors =
        this->globalDescriptorAllocator.allocate(this->_device, this->_drawImageDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, this->_drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(this->_device, this->_drawImageDescriptors);
    }

    // Clean up the descriptor allocator and new layout
    this->_mainDeletionQueue.push_function([&, this]() {
        this->globalDescriptorAllocator.destroy_pools(this->_device);
        vkDestroyDescriptorSetLayout(this->_device, this->_drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(this->_device, this->_singleImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(this->_device, this->_gpuSceneDataDescriptorLayout, nullptr);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        };

        this->_frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        this->_frames[i]._frameDescriptors.init(this->_device, 1000, frame_sizes);

        this->_mainDeletionQueue.push_function(
            [&, i, this]() { this->_frames[i]._frameDescriptors.destroy_pools(this->_device); });
    }
}

void VulkanEngine::init_pipelines() {
    this->init_background_pipelines();
    this->init_triangle_pipeline();
    this->init_mesh_pipeline();

    this->metalRoughMaterial.build_pipelines(this);
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
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &this->_singleImageDescriptorLayout;

    VK_CHECK(vkCreatePipelineLayout(this->_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

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
        utils::set_vk_object_debug_name(this->_device, (uint64_t)gradient.pipeline, VK_OBJECT_TYPE_PIPELINE,
                                        "gradient");
    }

    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky{
        .name = "sky", .layout = this->_gradientPipelineLayout, .data = {.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97)}};

    if (vkCreateComputePipelines(this->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr,
                                 &sky.pipeline) == VK_SUCCESS) {
        fmt::println("Sky pipeline created");
        utils::set_vk_object_debug_name(this->_device, (uint64_t)sky.pipeline, VK_OBJECT_TYPE_PIPELINE, "sky");
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
    this->mainCamera.velocity = glm::vec3(0.f);
    this->mainCamera.position = glm::vec3(30.f, -00.f, -085.f);

    this->mainCamera.pitch = 0;
    this->mainCamera.yaw = 0;

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

    // 3 default textures, white, grey, black. 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    this->_whiteImage =
        this->create_image((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    this->_greyImage =
        this->create_image((void*)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    this->_blackImage =
        this->create_image((void*)&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels; // for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    this->_errorCheckerboardImage =
        this->create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &sampl, nullptr, &this->_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &this->_defaultSamplerLinear);

    this->_mainDeletionQueue.push_function([&, this]() {
        vkDestroySampler(this->_device, this->_defaultSamplerNearest, nullptr);
        vkDestroySampler(this->_device, this->_defaultSamplerLinear, nullptr);

        this->destroy_image(this->_whiteImage);
        this->destroy_image(this->_greyImage);
        this->destroy_image(this->_blackImage);
        this->destroy_image(this->_errorCheckerboardImage);
    });

    GLTFMetallic_Roughness::MaterialResources materialResources{.colorImage = this->_whiteImage,
                                                                .colorSampler = this->_defaultSamplerLinear,
                                                                .metalRoughImage = this->_whiteImage,
                                                                .metalRoughSampler = this->_defaultSamplerLinear};

    AllocatedBuffer materialConstants =
        this->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, "GLTF Metallic Roughness Buffer");

    // Write to the buffer
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData =
        (GLTFMetallic_Roughness::MaterialConstants*)materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{1, 1, 1, 1};
    sceneUniformData->metal_rough_factors = glm::vec4{1, 0.5, 0, 0};

    this->_mainDeletionQueue.push_function([=, this]() { this->destroy_buffer(materialConstants); });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;
    this->defaultData = metalRoughMaterial.write_material(this->_device, MaterialPass::MainColor, materialResources,
                                                          this->globalDescriptorAllocator);
}

void VulkanEngine::init_renderables() {
    char buffer[MAX_PATH];
    std::string structurePath = utils::get_relative_path(buffer, MAX_PATH, "assets\\structure.glb");
    auto structureFile = loadGLTF(this, structurePath);
    assert(structureFile.has_value());
    this->loadedScenes["structure"] = *structureFile;
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

bool is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    std::array<glm::vec3, 8> corners{
        glm::vec3{1, 1, 1},  glm::vec3{1, 1, -1},  glm::vec3{1, -1, 1},  glm::vec3{1, -1, -1},
        glm::vec3{-1, 1, 1}, glm::vec3{-1, 1, -1}, glm::vec3{-1, -1, 1}, glm::vec3{-1, -1, -1},
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min_value = {1.5, 1.5, 1.5};
    glm::vec3 max_value = {-1.5, -1.5, -1.5};

    for (int c = 0; c < 8; c++) {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min_value = glm::vec3(min(v.x, min_value.x), min(v.y, min_value.y), min(v.z, min_value.z));
        max_value = glm::vec3(max(v.x, max_value.x), max(v.y, max_value.y), max(v.z, max_value.z));
    }

    // check the clip space box is within the view
    if (min_value.z > 1.f || max_value.z < 0.f || min_value.x > 1.f || max_value.x < -1.f || min_value.y > 1.f ||
        max_value.y < -1.f) {
        return false;
    } else {
        return true;
    }
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

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(this->_device);
        this->loadedScenes.clear();

        // Free the command pool
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            FrameData& frameData = this->_frames[i];
            // vkDestroyCommandPool(this->_device, frameData._commandPool, nullptr);

            // // Destroy the sync objects
            // vkDestroyFence(this->_device, frameData._renderFence, nullptr);
            // vkDestroySemaphore(this->_device, frameData._renderSemaphore, nullptr);
            // vkDestroySemaphore(this->_device, frameData._swapchainSemaphore, nullptr);

            frameData._deletionQueue.flush();
        }

        metalRoughMaterial.clear_resources(this->_device);
        this->_mainDeletionQueue.flush();

        // Perform all of the clean up operations
        this->destroy_swapchain();

        vkDestroySurfaceKHR(this->_instance, this->_surface, nullptr);
        fmt::println("Destroying device");
        vkDestroyDevice(this->_device, nullptr);

        vkb::destroy_debug_utils_messenger(this->_instance, this->_debugMessenger);
        vkDestroyInstance(this->_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw() {
    this->update_scene();

    VK_CHECK(vkWaitForFences(this->_device, 1, &this->get_current_frame()._renderFence, true, 1000000000));
    // Flush the deletion queue
    this->get_current_frame()._deletionQueue.flush();
    this->get_current_frame()._frameDescriptors.clear_pools(this->_device);

    // Wait until the GPU finished rendering the last frame, with a timeout of 1
    // second
    VK_CHECK(vkWaitForFences(this->_device, 1, &this->get_current_frame()._renderFence, true, 1000000000));

    uint32_t swapchainImageIndex;
    // When acquiring the image from the swapchain, we request an available one.
    // We have a timeout to wait until the next image is available if there is
    // none available.
    VkResult err =
        vkAcquireNextImageKHR(this->_device, this->_swapchain, 1000000000,
                              this->get_current_frame()._swapchainSemaphore, VK_NULL_HANDLE, &swapchainImageIndex);

    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        this->resize_requested = true;
        return;
    }

    this->_drawExtent.width = min(this->_swapchainExtent.width, this->_drawImage.imageExtent.width) * this->renderScale;
    this->_drawExtent.height =
        min(this->_swapchainExtent.height, this->_drawImage.imageExtent.height) * this->renderScale;

    VK_CHECK(vkResetFences(this->_device, 1, &this->get_current_frame()._renderFence));

    VkCommandBuffer cmd = this->get_current_frame()._mainCommandBuffer;
    // Ensure that our cmd buffer is resetted
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin recording, we use the command buffer exactly once
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Begin recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition the main draw image into a general layout to write to it
    // Then overwrite it all, b/c we dont care about what the older layout was
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    this->draw_background(cmd);

    // For start, transition the swapchain image to a drawable layout, then
    // clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, this->_depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    this->draw_geometry(cmd);

    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
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
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                   this->get_current_frame()._swapchainSemaphore);

    // The render semaphore signals that the rendering finished
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR,
                                                                     this->get_current_frame()._renderSemaphore);

    // Submit the command buffer to the queue and execute it
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // Block the render fence until the graphics commands finish executing
    VK_CHECK(vkQueueSubmit2(this->_graphicsQueue, 1, &submit, this->get_current_frame()._renderFence));

    // Present the image

    VkPresentInfoKHR presentInfo = vkinit::present_info();

    // we have a single swapchain
    presentInfo.pSwapchains = &this->_swapchain;
    presentInfo.swapchainCount = 1;

    // we have a wait semaphore
    presentInfo.pWaitSemaphores = &this->get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(this->_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }

    this->_frameNumber++;
}

void VulkanEngine::drawv2() {
    VK_CHECK(vkWaitForFences(this->_device, 1, &this->get_current_frame()._renderFence, true, 1000000000));

    FrameData* frame = &this->get_current_frame();

    frame->_deletionQueue.flush();
    frame->_frameDescriptors.clear_pools(this->_device);

    uint32_t swapchainImageIndex;

    VkResult e = vkAcquireNextImageKHR(this->_device, this->_swapchain, 1000000000, frame->_swapchainSemaphore, nullptr,
                                       &swapchainImageIndex);

    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        this->resize_requested = true;
        return;
    }

    this->_drawExtent.height =
        min(this->_swapchainExtent.height, this->_drawImage.imageExtent.height) * this->renderScale;
    this->_drawExtent.width = min(this->_swapchainExtent.width, this->_drawImage.imageExtent.width) * this->renderScale;

    VK_CHECK(vkResetFences(this->_device, 1, &frame->_renderFence));
    VK_CHECK(vkResetCommandBuffer(frame->_mainCommandBuffer, 0));

    VkCommandBuffer cmd = frame->_mainCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    this->draw_background(cmd);

    // For start, transition the swapchain image to a drawable layout, then
    // clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, this->_depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // TODO: Draw the geometry
    this->draw_geometry(cmd);

    // For start, transition the swapchain image to a drawable layout, then
    // clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, this->_depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // > Begin drawing imgui
    // Execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, this->_drawImage.image, this->_swapchainImages[swapchainImageIndex],
                                this->_drawExtent, this->_swapchainExtent);

    // Set the swapchain image layout to present so we can show it on the
    // screen.
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    this->draw_imgui(cmd, this->_swapchainImageViews[swapchainImageIndex]);

    // set the swapchain image layout to present to actually be able to show it on screen
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame->_swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR, frame->_renderSemaphore);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(this->_graphicsQueue, 1, &submit, frame->_renderFence));

    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        this->resize_requested = true;
        return;
    }
    // increase the number of frames drawn
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

void VulkanEngine::draw_main(VkCommandBuffer cmd) {
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {
    VkClearColorValue clearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

    ComputeEffect& effect = this->backgroundEffects[this->currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->_gradientPipelineLayout, 0, 1,
                            &this->_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, this->_gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants),
                       &effect.data);

    vkCmdDispatch(cmd, std::ceil(this->_windowExtent.width / 16.0), std::ceil(this->_windowExtent.height) / 16.0, 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {
    // Reset the counters
    this->stats.drawcall_count = 0;
    this->stats.triangle_count = 0;
    auto start = std::chrono::system_clock::now();

    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(this->drawCommands.OpaqueSurfaces.size());

    for (uint32_t i = 0; i < drawCommands.OpaqueSurfaces.size(); i++) {
        if (is_visible(drawCommands.OpaqueSurfaces[i], sceneData.viewproj)) {
            opaque_draws.push_back(i);
        }
    }

    std::sort(opaque_draws.begin(), opaque_draws.end(), [&, this](const auto& iA, const auto& iB) {
        const RenderObject& A = this->drawCommands.OpaqueSurfaces[iA];
        const RenderObject& B = this->drawCommands.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            return A.index_buffer < B.index_buffer;
        } else {
            return A.material < B.material;
        }
    });

    // begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(this->_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment =
        vkinit::depth_attachment_info(this->_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(this->_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, this->_trianglePipeline);

    // set dynamic viewport and scissor
    VkViewport viewport = {.x = 0,
                           .y = 0,
                           .width = (float)this->_drawExtent.width,
                           .height = (float)this->_drawExtent.height,
                           .minDepth = 0.0f,
                           .maxDepth = 1.0f};

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {.offset = VkOffset2D{.x = 0, .y = 0},
                        .extent = VkExtent2D{.width = (uint32_t)viewport.width, .height = (uint32_t)viewport.height}};

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // launch a draw command to draw 3 vertices
    // vkCmdDraw(cmd, 3, 1, 0, 0);

    AllocatedBuffer gpuSceneDataBuffer = this->create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                             VMA_MEMORY_USAGE_CPU_TO_GPU, "GPU Scene Data");

    this->get_current_frame()._deletionQueue.push_function([=, this]() { this->destroy_buffer(gpuSceneDataBuffer); });

    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;

    VkDescriptorSet globalDescriptor =
        this->get_current_frame()._frameDescriptors.allocate(this->_device, this->_gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(this->_device, globalDescriptor);

    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&, this](const RenderObject& r) {
        if (r.material != lastMaterial) {
            lastMaterial = r.material;
            // rebind pipeline and descriptors if the material changed
            if (r.material->pipeline != lastPipeline) {
                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1,
                                        &globalDescriptor, 0, nullptr);

                VkViewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = (float)_windowExtent.width;
                viewport.height = (float)_windowExtent.height;
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;

                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _windowExtent.width;
                scissor.extent.height = _windowExtent.height;

                vkCmdSetScissor(cmd, 0, 1, &scissor);
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1,
                                    &r.material->material_set, 0, nullptr);
        }
        // rebind index buffer if needed
        if (r.index_buffer != lastIndexBuffer) {
            lastIndexBuffer = r.index_buffer;
            vkCmdBindIndexBuffer(cmd, r.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        }
        // calculate final mesh matrix
        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertex_buffer_address;

        vkCmdPushConstants(cmd, r.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(cmd, r.index_count, 1, r.first_index, 0, 0);
        // stats
        stats.drawcall_count++;
        stats.triangle_count += r.index_count / 3;
    };

    for (auto& r : opaque_draws) {
        draw(this->drawCommands.OpaqueSurfaces[r]);
    }

    for (auto& r : this->drawCommands.TransparentSurfaces) {
        draw(r);
    }

    vkCmdEndRendering(cmd);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.f;
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        auto start = std::chrono::system_clock::now();
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
            this->mainCamera.processSDLEvent(e);
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

        ImGui::Begin("Stats");

        ImGui::Text("Frame Time: %f ms", stats.frametime);
        ImGui::Text("Draw Time: %f ms", stats.mesh_draw_time);
        ImGui::Text("Update Time: %f ms", stats.scene_update_time);
        ImGui::Text("Triangles %i ", stats.triangle_count);
        ImGui::Text("Draws %i ", stats.drawcall_count);

        ImGui::End();

        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &this->renderScale, 0.3, 1.0f);
            ComputeEffect& selected = this->backgroundEffects[this->currentBackgroundEffect];
            ImGui::Text("Selected effect: %s", selected.name);
            ImGui::SliderInt("Effect Index", &this->currentBackgroundEffect, 0, this->backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);
            ImGui::InputFloat4("data3", (float*)&selected.data.data3);
            ImGui::InputFloat4("data4", (float*)&selected.data.data4);
        }
        ImGui::End();

        ImGui::Render();

        this->update_scene();
        this->draw();

        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
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

    AllocatedBuffer newBuffer{.name = name};
    VK_CHECK(vmaCreateBuffer(this->_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation,
                             &newBuffer.info));

    utils::set_vk_object_debug_name(this->_device, (uint64_t)newBuffer.buffer, VK_OBJECT_TYPE_BUFFER, name);
    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer) {
#if PRINT
    fmt::println("Destroying buffer: {} at address: {:#016x}", buffer.name, (uint64_t)buffer.buffer);
#endif
    vmaDestroyBuffer(this->_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer =
        this->create_buffer(vertexBufferSize,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            VMA_MEMORY_USAGE_GPU_ONLY, "vertex_buffer");
    VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                .buffer = newSurface.vertexBuffer.buffer};
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(this->_device, &deviceAddressInfo);

    newSurface.indexBuffer =
        this->create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_GPU_ONLY, "index_buffer");

    AllocatedBuffer staging = this->create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped) {
    AllocatedImage newImage{
        .imageExtent = size,
        .imageFormat = format,
    };

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {.usage = VMA_MEMORY_USAGE_GPU_ONLY,
                                         .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

    // allocate and create the image
    VK_CHECK(vmaCreateImage(this->_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(this->_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                          bool mipmapped) {
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer =
        this->create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, "Upload Buffer");
    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = this->create_image(
        size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    this->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {.bufferOffset = 0,
                                        .bufferRowLength = 0,
                                        .bufferImageHeight = 0,
                                        .imageSubresource =
                                            {
                                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                .mipLevel = 0,
                                                .baseArrayLayer = 0,
                                                .layerCount = 1,
                                            },
                                        .imageExtent = size};

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copyRegion);

        if (mipmapped) {
            vkutil::generate_mipmaps(cmd, new_image.image,
                                     VkExtent2D{new_image.imageExtent.width, new_image.imageExtent.height});
        } else {
            vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
    this->destroy_buffer(uploadbuffer);
    return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage& img) {
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine) {
    VkShaderModule meshFragShader;

    char buffer[MAX_PATH];
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\default_frag.spv"),
                                    engine->_device, &meshFragShader)) {
        fmt::println("Error when building the triangle fragment shader module!");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module(utils::get_relative_path(buffer, MAX_PATH, "shaders\\default_vert.spv"),
                                    engine->_device, &meshVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module!");
    }

    VkPushConstantRange matrixRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants),
    };

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_bindings(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_bindings(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_bindings(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    this->materialLayout =
        layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = {engine->_gpuSceneDataDescriptorLayout, this->materialLayout};

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules
    // per stage
    vkutil::PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader)
        .set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .set_polygon_mode(VK_POLYGON_MODE_FILL)
        .set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
        .set_multisampling_none()
        .disable_blending()
        .enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL)
        .set_color_attachment_format(engine->_drawImage.imageFormat)
        .set_depth_format(engine->_depthImage.imageFormat);

    pipelineBuilder._pipelineLayout = newLayout;

    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device, "Opaque");

    // Create the transparent values
    transparentPipeline.pipeline = pipelineBuilder.enable_blending_additive()
                                       .enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL)
                                       .build_pipeline(engine->_device, "Transparent");

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

void GLTFMetallic_Roughness::clear_resources(VkDevice device) {
    vkDestroyDescriptorSetLayout(device, this->materialLayout, nullptr);

    fmt::println("Destroying transparent pipeline");
    vkDestroyPipelineLayout(device, this->transparentPipeline.layout, nullptr);

    vkDestroyPipeline(device, this->transparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, this->opaquePipeline.pipeline, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass,
                                                        const MaterialResources& resources,
                                                        DescriptorAllocatorGrowable& descriptorAllocator) {
    MaterialInstance matData;
    matData.pass_type = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    } else {
        matData.pipeline = &opaquePipeline;
    }

    matData.material_set = descriptorAllocator.allocate(device, this->materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(device, matData.material_set);

    return matData;
}

void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx) {
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : this->mesh->surfaces) {
        RenderObject def{.index_count = s.count,
                         .first_index = s.start_index,
                         .index_buffer = this->mesh->meshBuffers.indexBuffer.buffer,
                         .material = &s.material->data,
                         .bounds = s.bounds,
                         .transform = nodeMatrix,
                         .vertex_buffer_address = this->mesh->meshBuffers.vertexBufferAddress};

        if (s.material->data.pass_type == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        } else {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    Node::Draw(topMatrix, ctx);
}

void VulkanEngine::update_scene() {
    this->mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();
    glm::mat4 proj = glm::perspective(
        glm::radians(70.f), (float)this->_windowExtent.width / (float)this->_windowExtent.height, 0.1f, 1000.f);

    proj[1][1] *= -1;
    this->sceneData.view = view;
    this->sceneData.proj = proj;
    this->sceneData.viewproj = proj * view;
    // some default lighting parameters
    this->sceneData.ambientColor = glm::vec4(.1f);
    this->sceneData.sunlightColor = glm::vec4(1.f);
    this->sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);

    this->drawCommands.OpaqueSurfaces.clear();
    this->drawCommands.TransparentSurfaces.clear();
    this->loadedScenes["structure"]->Draw(glm::mat4{1.f}, this->drawCommands);
}
