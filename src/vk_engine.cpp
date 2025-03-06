//> includes
#include "vk_engine.h"

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
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
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

    VkPhysicalDeviceVulkan13Features features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features.dynamicRendering = true;
    features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features1_2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features1_2.bufferDeviceAddress = true;
    features1_2.descriptorIndexing = true;
    vkb::PhysicalDeviceSelector selector{vkb_inst};

    vkb::PhysicalDevice physicalDevice = selector
                                             .set_minimum_version(1, 3)
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

    this->_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    this->_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
}

void VulkanEngine::init_swapchain()
{
    this->create_swapchain(_windowExtent.width, _windowExtent.height);
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{this->_chosenGPU, this->_device, this->_surface};
    _swapchainImageFormat = VK_FORMAT_B8G8R8_UNORM; // 32 bit format

    vkb::Swapchain vkbSwapchain = swapchainBuilder
                                      .set_desired_format(VkSurfaceFormatKHR{.format = this->_swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
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

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(this->_device, this->_swapchain, nullptr);

    for (int i = 0; i < _swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(this->_device, this->_swapchainImageViews[i], nullptr);
    }
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.pNext = nullptr;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // We say that the command buffer is compatible with the graphics queue family
    commandPoolInfo.queueFamilyIndex = this->_graphicsQueueFamily;

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(this->_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
        // Allocate the default cmd buffer to use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(this->_device, &cmdAllocInfo, &this->_frames[i]._mainCommandBuffer));
    }
}

void VulkanEngine::init_sync_structures()
{
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
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        vkDeviceWaitIdle(this->_device);

        // Free the command pool
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            vkDestroyCommandPool(this->_device, this->_frames[i]._commandPool, nullptr);
        }
        // Perform all of the clean up operations
        destroy_swapchain();

        vkDestroySurfaceKHR(this->_instance, this->_surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(this->_instance, this->_debugMessenger);
        vkDestroyInstance(this->_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    FrameData& frameData = get_current_frame();
    VkDevice device = this->_device;

    // Wait until the GPU finished rendering the last frame, with a timeout of 1 second
    VK_CHECK(vkWaitForFences(device, 1, &frameData._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(device, 1, &frameData._renderFence));

    uint32_t swapchainImageIndex;
    // When acquiring the image from the swapchain, we request an available one. We have a timeout to wait
    // until the next image is available if there is none available.
    VK_CHECK(vkAcquireNextImageKHR(device, this->_swapchain, 1000000000, frameData._swapchainSemaphore, nullptr, &swapchainImageIndex));

    VkCommandBuffer cmd = frameData._mainCommandBuffer;
    // Ensure that our cmd buffer is resetted
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin recording, we use the command buffer exactly once
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Begin recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // For start, transition the swapchain image to a drawable layout, then clear it, and finally transition it back for display
    vkutil::transition_image(cmd, this->_swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit)
    {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0)
        {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT)
            {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED)
                {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED)
                {
                    stop_rendering = false;
                }
            }

            if (e.type == SDL_KEYUP)
            {
                fmt::println("Released: {}", e.key.keysym.sym);
                fmt::println("Key up detected");
            }
        }

        // do not draw if we are minimized
        if (stop_rendering)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}