//> includes
#include <vector>
#include <vulkan/vulkan_core.h>
#define VMA_IMPLEMENTATION
#include "vk_engine.h"
#include "vk_mem_alloc.h"
#include "vk_types.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"
#include "vk_images.h"

VulkanEngine *loadedEngine = nullptr;

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init() {
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;

    // Use the debugger
    auto inst = builder.set_app_name("Vk Renderer")
                    .request_validation_layers(true)
                    .use_default_debug_messenger()
                    .require_api_version(1, 3, 0)
                    .build();

    vkb::Instance vkb_inst = inst.value();
    this->_instance = vkb_inst.instance;
    this->_debugMessenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(this->_window, this->_instance, &_surface);

    VkPhysicalDeviceVulkan13Features features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features.dynamicRendering = true;
    features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features1_2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features1_2.bufferDeviceAddress = true;
    features1_2.descriptorIndexing = true;
    vkb::PhysicalDeviceSelector selector{vkb_inst};

    vkb::PhysicalDevice physicalDevice =
        selector.set_minimum_version(1, 3)
            .set_required_features_13(features)
            .set_required_features_12(features1_2)
            .set_surface(this->_surface)
            .select()
            .value();

    // Create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of the vulkan application
    this->_device = vkbDevice.device;
    this->_chosenGPU = physicalDevice.physical_device;

    this->_graphicsQueue =
        vkbDevice.get_queue(vkb::QueueType::graphics).value();
    this->_graphicsQueueFamily =
        vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = this->_chosenGPU;
    allocatorInfo.device = this->_device;
    allocatorInfo.instance = this->_instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &this->_allocator);

    this->_mainDeletionQueue.push_function([&]() {
        fmt::println("Releasing VMA");
        vmaDestroyAllocator(this->_allocator);
    });
}

void VulkanEngine::init_swapchain() {
    this->create_swapchain(_windowExtent.width, _windowExtent.height);
    VkExtent3D drawImageExtent = {this->_windowExtent.width,
                                  this->_windowExtent.height, 1};

    // 64 bits per pixel
    this->_drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    this->_drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimgInfo = vkinit::image_create_info(
        this->_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // Allocate the image on gpu local memory
    VmaAllocationCreateInfo rimgAllocInfo = {};
    rimgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimgAllocInfo.requiredFlags =
        VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vmaCreateImage(this->_allocator, &rimgInfo, &rimgAllocInfo,
                   &_drawImage.image, &_drawImage.allocation, VK_NULL_HANDLE);

    // Allocate and create the image
    VkImageViewCreateInfo rViewInfo = vkinit::imageview_create_info(
        this->_drawImage.imageFormat, this->_drawImage.image,
        VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(this->_device, &rViewInfo, VK_NULL_HANDLE,
                               &this->_drawImage.imageView));

    // Add to the deletion queue because we've allocated memory on the gpu
    this->_mainDeletionQueue.push_function([=, this]() {
        vkDestroyImageView(this->_device, this->_drawImage.imageView,
                           VK_NULL_HANDLE);
        vmaDestroyImage(this->_allocator, this->_drawImage.image,
                        this->_drawImage.allocation);
    });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{this->_chosenGPU, this->_device,
                                           this->_surface};
    _swapchainImageFormat = VK_FORMAT_B8G8R8_UNORM; // 32 bit format

    vkb::Swapchain vkbSwapchain =
        swapchainBuilder
            .set_desired_format(VkSurfaceFormatKHR{
                .format = this->_swapchainImageFormat,
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

    for (int i = 0; i < _swapchainImageViews.size(); i++) {
        vkDestroyImageView(this->_device, this->_swapchainImageViews[i],
                           nullptr);
    }
}

void VulkanEngine::init_commands() {
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.pNext = nullptr;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // We say that the command buffer is compatible with the graphics queue
    // family
    commandPoolInfo.queueFamilyIndex = this->_graphicsQueueFamily;

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(this->_device, &commandPoolInfo, nullptr,
                                     &_frames[i]._commandPool));
        // Allocate the default cmd buffer to use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo =
            vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(
            vkAllocateCommandBuffers(this->_device, &cmdAllocInfo,
                                     &this->_frames[i]._mainCommandBuffer));
    }
}

void VulkanEngine::init_sync_structures() {
    // First fence controls when the gpu finished rendering the frame
    // 2 semaphore to synchronize rendering with swapchain
    // The fence signals so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo =
        vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        FrameData &frameData = this->_frames[i];
        VK_CHECK(vkCreateFence(this->_device, &fenceCreateInfo, nullptr,
                               &frameData._renderFence));

        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr,
                                   &frameData._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(this->_device, &semaphoreCreateInfo, nullptr,
                                   &frameData._renderSemaphore));
    }
}

void VulkanEngine::init_descriptors() {
    // Create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};

    this->globalDescriptorAllocator.init_pool(this->_device, 10, sizes);

    // Make the descriptor set layout for our compute draw, bind to the first
    // layer
    DescriptorLayoutBuilder builder;
    builder = builder.add_bindings(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    this->_drawImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    // Allocate a descriptor for our draw image
    this->_drawImageDescriptors = this->globalDescriptorAllocator.allocate(
        this->_device, this->_drawImageDescriptorLayout);

    VkDescriptorImageInfo imgInfo{
        .imageView = this->_drawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    VkWriteDescriptorSet drawImageWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = this->_drawImageDescriptors,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &imgInfo
    };

    vkUpdateDescriptorSets(this->_device, 1, &drawImageWrite, 0, nullptr);
    
    // Clean up the descriptor allocator and new layout
    this->_mainDeletionQueue.push_function([&]() {
        this->globalDescriptorAllocator.destroy_pool(this->_device);
        vkDestroyDescriptorSetLayout(this->_device, this->_drawImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(this->_device);

        // Free the command pool
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            FrameData frameData = this->_frames[i];
            vkDestroyCommandPool(this->_device, frameData._commandPool,
                                 nullptr);

            // Destroy the sync objects
            vkDestroyFence(this->_device, frameData._renderFence, nullptr);
            vkDestroySemaphore(this->_device, frameData._renderSemaphore,
                               nullptr);
            vkDestroySemaphore(this->_device, frameData._swapchainSemaphore,
                               nullptr);
        }
        this->_mainDeletionQueue.flush();

        // Perform all of the clean up operations
        destroy_swapchain();

        vkDestroySurfaceKHR(this->_instance, this->_surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(this->_instance,
                                           this->_debugMessenger);
        vkDestroyInstance(this->_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw() {
    FrameData &frameData = get_current_frame();
    VkDevice device = this->_device;

    // Wait until the GPU finished rendering the last frame, with a timeout of 1
    // second
    VK_CHECK(
        vkWaitForFences(device, 1, &frameData._renderFence, true, 1000000000));

    // Flush the deletion queue
    frameData._deletionQueue.flush();

    VK_CHECK(vkResetFences(device, 1, &frameData._renderFence));

    uint32_t swapchainImageIndex;
    // When acquiring the image from the swapchain, we request an available one.
    // We have a timeout to wait until the next image is available if there is
    // none available.
    VK_CHECK(vkAcquireNextImageKHR(device, this->_swapchain, 1000000000,
                                   frameData._swapchainSemaphore,
                                   VK_NULL_HANDLE, &swapchainImageIndex));

    VkCommandBuffer cmd = frameData._mainCommandBuffer;
    // Ensure that our cmd buffer is resetted
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin recording, we use the command buffer exactly once
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    this->_drawExtent.width = this->_drawImage.imageExtent.width;
    this->_drawExtent.height = this->_drawImage.imageExtent.height;

    // Begin recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // Transition the main draw image into a general layout to write to it
    // Then overwrite it all, b/c we dont care about what the older layout was
    vkutil::transition_image(cmd, this->_drawImage.image,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    this->draw_background(cmd);

    // For start, transition the swapchain image to a drawable layout, then
    // clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_drawImage.image,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Change the swapchain back to a format that works for displays
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex],
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, this->_drawImage.image,
                                this->_swapchainImages[swapchainImageIndex],
                                this->_drawExtent, this->_swapchainExtent);

    // Set the swapchain image layout to present so wecan show it on the screen.
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex],
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Prepare the submission to the queue
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    // Need to wait until the presentSemaphore (this will tell us that the
    // swapchain is ready)
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        frameData._swapchainSemaphore);

    // The render semaphore signals that the rendering finished
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR, frameData._renderSemaphore);

    // Submit the command buffer to the queue and execute it
    VkSubmitInfo2 submit =
        vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // Block the render fence until the graphics commands finish executing
    VK_CHECK(vkQueueSubmit2(this->_graphicsQueue, 1, &submit,
                            frameData._renderFence));

    // Present the image
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    // we have a single swapchain
    presentInfo.pSwapchains = &this->_swapchain;
    presentInfo.swapchainCount = 1;

    // we have a wait semaphore
    presentInfo.pWaitSemaphores = &frameData._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(this->_graphicsQueue, &presentInfo));

    this->_frameNumber++;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {
    VkClearColorValue clearValue;
    float flash = std::abs(std::sin(this->_frameNumber / 120.f));
    clearValue = {{0.0f, 0.0f, flash, 1.0f}};

    VkImageSubresourceRange clearRange =
        vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(cmd, this->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clearValue, 1, &clearRange);
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
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}
