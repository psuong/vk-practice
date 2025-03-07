// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "deletion_queue.h"
#include "vk_descriptors.h"
#include "vk_mem_alloc.h"
#include <vk_types.h>

constexpr unsigned int FRAME_OVERLAP = 2;

struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct FrameData {
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	// The swapchain semaphore is used for that render commands wait on the swapchain image request.
	VkSemaphore _swapchainSemaphore;

	// The renderSemaphore controls presenting the image to the OS once the drawing finishes
	VkSemaphore _renderSemaphore;
	VkFence _renderFence;

    DeletionQueue _deletionQueue;
};

class VulkanEngine
{
public:

	bool _isInitialized{false};
	int _frameNumber{0};
	bool stop_rendering{false};
	VkExtent2D _windowExtent{1700, 900};

	// Forward declaration without having to include the SDL header
	struct SDL_Window *_window{nullptr};

	FrameData _frames[FRAME_OVERLAP];

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	static VulkanEngine &Get();

	DescriptorAllocator globalDescriptorAllocator;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	// initializes everything in the engine
	void init();

	// shuts down the engine
	void cleanup();

	// draw loop
	void draw();

	// run main loop
	void run();

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
private:
	VkInstance _instance;					    // Vulkan library handle
	VkDebugUtilsMessengerEXT _debugMessenger;   // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU;			    // GPU chosen as the default device
	VkDevice _device;						    // Vulkan device for commands
	VkSurfaceKHR _surface;					    // Vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator;
    AllocatedImage _drawImage;
    VkExtent2D _drawExtent;

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
    void init_descriptors();
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

    void draw_background(VkCommandBuffer cmd);
};
