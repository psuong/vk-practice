#pragma once

#include <cstring>
#include <string.h>
#include <windows.h>

namespace utils {
const char *get_shader_path(char *buffer, size_t bufferSize,
                            const char *subdir) {
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    char *lastSlash = strrchr(buffer, '\\');

    if (lastSlash) {
        *lastSlash = '\0';
    }

    strcat_s(buffer, bufferSize, "\\");
    strcat_s(buffer, bufferSize, subdir);
    return buffer;
}
} // namespace utils
