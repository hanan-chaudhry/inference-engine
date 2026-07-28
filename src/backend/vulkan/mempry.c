#include "../../../inc/backend/vk_mem.h"
#include <stdio.h>
#include <stdlib.h>

//find a valid memory type supported by the physical device
static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    printf("ERROR: Failed to find suitable memory type!\n");
    return 0;
}

bool vk_allocate_tensor(VkContext* ctx, VkTensorBuffer* tensor, size_t size) {
    tensor->size = size;

    // 1. Create the buffer
    VkBufferCreateInfo buffer_info = { 0 };
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size; // Logical math size
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (vkCreateBuffer(ctx->device, &buffer_info, NULL, &tensor->buffer) != VK_SUCCESS) {
        printf("ERROR: Failed to create buffer.\n");
        return false;
    }

    // 2. Check hardware alignment and size requirements
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, tensor->buffer, &mem_reqs);

    // 3. Allocate memory 
    VkMemoryAllocateInfo alloc_info = { 0 };
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        ctx->physical_device,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (vkAllocateMemory(ctx->device, &alloc_info, NULL, &tensor->memory) != VK_SUCCESS) {
        printf("ERROR: Failed to allocate device memory.\n");
        return false;
    }

    vkBindBufferMemory(ctx->device, tensor->buffer, tensor->memory, 0);

    // 4. Map the memory so CPU can read/write to it easily
    if (vkMapMemory(ctx->device, tensor->memory, 0, size, 0, &tensor->mapped_ptr) != VK_SUCCESS) {
        printf("ERROR: Failed to map tensor memory.\n");
        return false;
    }

    return true;
}

void vk_free_tensor(VkContext* ctx, VkTensorBuffer* tensor) {
    if (!tensor) return;
    if (tensor->memory) {
        vkUnmapMemory(ctx->device, tensor->memory);
    }
    if (tensor->buffer) {
        vkDestroyBuffer(ctx->device, tensor->buffer, NULL);
    }
    if (tensor->memory) {
        vkFreeMemory(ctx->device, tensor->memory, NULL);
    }
}