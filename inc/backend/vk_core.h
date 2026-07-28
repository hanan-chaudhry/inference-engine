#ifndef VK_CORE_H
#define VK_CORE_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdbool.h>

// holds connection to the GPU.
typedef struct {
    VkInstance instance;                 // vulkan instance
    VkPhysicalDevice physical_device;    // The physical silicon (e.g., your AMD GPU)
    VkDevice device;                     // The software connection to the GPU
    VkQueue compute_queue;               // The mailbox where you send commands
    uint32_t queue_family_index;         // The ID of the compute mailbox
    VkCommandPool command_pool;          // The memory pool used to write your "to-do lists"
} VkContext;

// Initializes the Vulkan context
bool vk_init(VkContext* ctx);

// Cleans up resources
void vk_cleanup(VkContext* ctx);

// Reads spv into buffer
char* vk_read_spv(const char* filename, size_t* file_length);

// Creates one universal blueprint for all shaders
bool vk_create_universal_layout(
    VkContext* ctx,
    VkDescriptorSetLayout* out_desc_layout,
    VkPipelineLayout* out_pipe_layout
);

VkPipeline vk_create_compute_pipeline(
    VkContext* ctx,
    const char* spv_path,
    VkPipelineLayout layout
);

#endif 