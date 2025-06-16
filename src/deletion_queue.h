#pragma once

#include <deque>
#include <functional>

/**
 * struct DeletionQueue - The purpose of the DeletionQueue is to allow us to scale and not have to
 * hard code and keep track of every Vulkan Handle.
 *
 * TODO Instead of storing functors, it makes more sense to store the VkHandle and free it when we
 * scale up.
 */
struct DeletionQueue {
    std::deque<std::function<void()>> deleters;

    void push_function(std::function<void()>&& function) {
        this->deleters.push_back(function);
    }

    void flush() {
        for (auto it = this->deleters.rbegin(); it != this->deleters.rend(); it++) {
            (*it)(); // Call the functor
        }
    }
};
