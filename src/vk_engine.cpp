﻿//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"

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
    vkb::DeviceBuilder deviceBuilder { physicalDevice };
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of the vulkan application
    this->_device = vkbDevice.device;
    this->_chosenGPU = physicalDevice.physical_device;
}

void VulkanEngine::init_swapchain()
{
    this->create_swapchain(_windowExtent.width, _windowExtent.height);
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{this->_chosenGPU, this->_device, this->_surface };
    _swapchainImageFormat = VK_FORMAT_B8G8R8_UNORM; // 32 bit format

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{.format = this->_swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
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

void VulkanEngine::init_commands()
{
}

void VulkanEngine::init_sync_structures()
{
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    // nothing yet
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