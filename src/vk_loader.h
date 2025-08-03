#pragma once

#include "vk_descriptors.h"
#include "vk_types.h"
#include <filesystem>
#include <unordered_map>
#include <vk_loader.h>

// Forward declaration
class VulkanEngine;

struct GLTFMaterial {
    MaterialInstance data;
};

struct GeoSurface {
    uint32_t start_index;
    uint32_t count;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

struct LoadedGLTF : public IRenderable {
    // storage for all the data on a given glTF file
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // nodes that dont have a parent, for iterating through the file in tree order
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine* creator;

    ~LoadedGLTF() {
        this->clearAll();
    };

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);

  private:
    void clearAll();
};

[[deprecated("Use loadGLTF instead")]]
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine,
                                                                      std::filesystem::path filePath);
std::optional<std::shared_ptr<LoadedGLTF>> loadGLTF(VulkanEngine* engine, std::string_view filePath);
