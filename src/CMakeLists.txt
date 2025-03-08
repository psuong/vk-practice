# Inside src/CMakeLists.txt

add_executable(engine 
  main.cpp
  vk_types.h
  vk_initializers.cpp
  vk_initializers.h
  vk_images.h
  vk_images.cpp 
  vk_descriptors.h
  vk_descriptors.cpp
  vk_pipelines.h
  vk_pipelines.cpp
  vk_engine.h
  vk_engine.cpp
  vk_loader.h
  vk_loader.cpp
  utils.h
  camera.cpp
  camera.h
  deletion_queue.h
)

# Set C++ standard
set_property(TARGET engine PROPERTY CXX_STANDARD 20)

# Add compile definitions (for GLM)
target_compile_definitions(engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

# Add include directories
target_include_directories(engine PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/third_party/SDL/include"  # Add SDL2 include directory, because clang is throwing errors (still compiles tho)
)

# Link libraries (including SDL2)
target_link_libraries(engine PUBLIC 
    vma 
    glm 
    Vulkan::Vulkan 
    fmt::fmt 
    stb_image 
    SDL2::SDL2
    vkbootstrap 
    imgui 
    fastgltf::fastgltf
)

# Set up precompiled headers
target_precompile_headers(engine PUBLIC 
    <optional> 
    <vector> 
    <memory> 
    <string> 
    <unordered_map> 
    <glm/mat4x4.hpp>  
    <glm/vec4.hpp> 
    <vulkan/vulkan.h>
)

# Copy DLLs after build (if needed)
add_custom_command(TARGET engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine>
  COMMAND_EXPAND_LISTS
)
