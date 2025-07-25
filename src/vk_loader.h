﻿#pragma once

#include "vk_types.h"
#include <vk_loader.h>
#include <unordered_map>
#include <filesystem>

struct GeoSurface {
    uint32_t start_index;
    uint32_t count;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

// Forward declaration
class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine,
                                                                      std::filesystem::path filePath);
