#pragma once

#include <cstring>
#include <string.h>
#include <windows.h>

namespace utils {
inline const char* get_relative_path(char* buffer, size_t bufferSize, const char* subdir) {
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    char* lastSlash = strrchr(buffer, '\\');

    if (lastSlash) {
        *lastSlash = '\0';
    }

    strcat_s(buffer, bufferSize, "\\");
    strcat_s(buffer, bufferSize, subdir);
    return buffer;
}

inline void set_pipeline_debug_name(VkDevice device, uint64_t objectHandle, VkObjectType objectType, const char* name) {
    PFN_vkSetDebugUtilsObjectNameEXT func =
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
    if (func) {
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = objectHandle;
        nameInfo.pObjectName = name;
        func(device, &nameInfo);
    }
}

} // namespace utils
