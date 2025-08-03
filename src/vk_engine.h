// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "camera.h"
#include "deletion_queue.h"
#include "glm/ext/vector_float4.hpp"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_mem_alloc.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

constexpr unsigned int FRAME_OVERLAP = 2;

struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char* name;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    ComputePushConstants data;
};

struct FrameData {
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    // The swapchain semaphore is used for that render commands wait on the
    // swapchain image request.
    VkSemaphore _swapchainSemaphore;

    // The renderSemaphore controls presenting the image to the OS once the
    // drawing finishes
    VkSemaphore _renderSemaphore;
    VkFence _renderFence;

    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};

struct RenderObject {
    uint32_t index_count;
    uint32_t first_index;
    VkBuffer index_buffer;

    MaterialInstance* material;
    glm::mat4 transform;
    VkDeviceAddress vertex_buffer_address;
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // padding, we need it anyway for uniform buffers
        glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine* engine);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources,
                                    DescriptorAllocatorGrowable& descriptorAllocator);
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
};

struct MeshNode : public Node {
    std::shared_ptr<MeshAsset> mesh;

  public:
    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

class VulkanEngine {
  public:
    Camera mainCamera;
    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{1600, 900};

    // Forward declaration without having to include the SDL header
    struct SDL_Window* _window{nullptr};

    FrameData _frames[FRAME_OVERLAP];

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    static VulkanEngine& Get();

    DescriptorAllocatorGrowable globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipelineLayout _gradientPipelineLayout;

    // Immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    // initializes everything in the engine
    void init();

    // shuts down the engine
    void cleanup();

    // draw loop
    void draw();

    // draw our ui
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    // run main loop
    void run();

    FrameData& get_current_frame() {
        return this->_frames[this->_frameNumber % FRAME_OVERLAP];
    };

    GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
    VkDevice _device; // Vulkan device for commands

    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage _drawImage;
    AllocatedImage _depthImage;

    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

    void update_scene();

  private:
    VkInstance _instance;                     // Vulkan library handle
    VkDebugUtilsMessengerEXT _debugMessenger; // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU;              // GPU chosen as the default device
    VkSurfaceKHR _surface;                    // Vulkan window surface

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator;
    VkExtent2D _drawExtent;
    float renderScale = 1.f;

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{0};

    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;

    GPUMeshBuffers rectangle;
    GPUSceneData sceneData;
    VkDescriptorSetLayout _singleImageDescriptorLayout;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    bool resize_requested;

    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void init_descriptors();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    void init_pipelines();
    void init_background_pipelines();
    void init_mesh_pipeline();

    void init_default_data();

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    void init_imgui();

    void init_triangle_pipeline();
    void draw_background(VkCommandBuffer cmd);
    void draw_geometry(VkCommandBuffer cmd);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                                  const char* name);
    void destroy_buffer(const AllocatedBuffer& buffer);
    void resize_swapchain();

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                bool mipmapped = false);
    void destroy_image(const AllocatedImage& img);
};
