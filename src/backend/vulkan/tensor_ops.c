#include <string.h>
#include <stdio.h>
#include "../../../inc/backend/vk_mem.h"
#include "../../../inc/backend/vk_ops.h"

void vk_dispatch(VkContext* ctx, VkPipeline pipeline, VkPipelineLayout layout,
    VkTensorBuffer** buffers, uint32_t buffer_count,
    PushParams* params, uint32_t group_x, uint32_t group_y, uint32_t group_z) {

    // 1. Transient Descriptor Pool
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer_count };
    VkDescriptorPoolCreateInfo pool_info = { 0 };
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool pool;
    vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &pool);

    // 2. Recreate the layout signature locally for allocation
    VkDescriptorSetLayoutBinding bindings[3] = { 0 };
    for (uint32_t i = 0; i < buffer_count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = { 0 };
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = buffer_count;
    layout_info.pBindings = bindings;

    VkDescriptorSetLayout temp_layout;
    vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &temp_layout);

    VkDescriptorSetAllocateInfo alloc_info = { 0 };
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &temp_layout;

    VkDescriptorSet descriptor_set;
    vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set);

    // 3. Write buffers dynamically
    VkDescriptorBufferInfo buffer_infos[3]; // Max 3 for now
    VkWriteDescriptorSet writes[3];
    for (uint32_t i = 0; i < buffer_count; i++) {
        buffer_infos[i].buffer = buffers[i]->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = buffers[i]->size;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = NULL;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, buffer_count, writes, 0, NULL);

    // 4. Command Buffer Execution
    VkCommandBufferAllocateInfo cmd_info = { 0 };
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = ctx->command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx->device, &cmd_info, &cmd);

    VkCommandBufferBeginInfo begin_info = { 0 };
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptor_set, 0, NULL);

    // Push our standardized struct
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushParams), params);

    // Dispatch 3D grid
    vkCmdDispatch(cmd, group_x, group_y, group_z);

    vkEndCommandBuffer(cmd);

    // 5. Submit & Sync
    VkSubmitInfo submit_info = { 0 };
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->compute_queue);

    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
    vkDestroyDescriptorSetLayout(ctx->device, temp_layout, NULL);
    vkDestroyDescriptorPool(ctx->device, pool, NULL);
}