#ifndef VK_MEM_H
#define VK_MEM_H

#include "vk_core.h"

typedef struct {
    VkBuffer buffer;            // label or type of buffer
    VkDeviceMemory memory;      // the actual VRAM allocation on GPU
    void* mapped_ptr;           // this is the pointer we write to and that data is pushed to the GPU. this is dest of data transfer in C
    size_t size;                // the size of data to be moved
} VkTensorBuffer;

// Allocates VRAM on the GPU
bool vk_allocate_tensor(VkContext* ctx, VkTensorBuffer* tensor, size_t size);

// Frees VRAM allocation
void vk_free_tensor(VkContext* ctx, VkTensorBuffer* tensor);

#endif 