// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>


template <>
struct fmt::formatter<VkResult> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    auto format(VkResult result, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}", fmt::string_view(string_VkResult(result)));
    }
};

#define VK_CHECK(x)                                                           \
    do {                                                                      \
        VkResult err = x;                                                     \
        if (err != VK_SUCCESS) {                                              \
            fmt::println("Vulkan error: {} at {}:{} - {}",                    \
                         err, __FILE__, __LINE__, string_VkResult(err));      \
            abort();                                                          \
        }                                                                     \
    } while (0)
