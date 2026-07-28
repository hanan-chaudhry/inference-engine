#include "../../../inc/backend/vk_core.h"
#include "../../../inc/backend/vk_ops.h"
#include <stdio.h>
#include <stdlib.h>

// read spv into buffer
char* vk_read_spv(const char* filename, size_t* file_length) {
    // Open in read binary mode
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("ERROR: Failed to open shader file %s\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *file_length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* buffer = malloc(*file_length);
    if (!buffer) {
        printf("ERROR: Failed to allocate memory for %s\n", filename);
        fclose(fp);
        return NULL;
    }

    fread(buffer, 1, *file_length, fp);
    fclose(fp);

    return buffer;
}

// compile the shader and build the pipieline
bool vk_create_universal_layout(VkContext* ctx, VkDescriptorSetLayout* out_desc_layout, VkPipelineLayout* out_pipe_layout) {
    // 3 Buffer Slots (Input A, Input B, Output C)
    VkDescriptorSetLayoutBinding bindings[3] = { 0 };
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo desc_info = { 0 };
    desc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desc_info.bindingCount = 3;
    desc_info.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx->device, &desc_info, NULL, out_desc_layout);

    // 1 Standardized Push Constant
    VkPushConstantRange push_constant = { 0 };
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(PushParams);

    VkPipelineLayoutCreateInfo pipe_info = { 0 };
    pipe_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_info.setLayoutCount = 1;
    pipe_info.pSetLayouts = out_desc_layout;
    pipe_info.pushConstantRangeCount = 1;
    pipe_info.pPushConstantRanges = &push_constant;

    return vkCreatePipelineLayout(ctx->device, &pipe_info, NULL, out_pipe_layout) == VK_SUCCESS;
}

VkPipeline vk_create_compute_pipeline(VkContext* ctx, const char* spv_path, VkPipelineLayout layout) {
    size_t file_size;
    char* code = vk_read_spv(spv_path, &file_size);
    if (!code) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo module_info = { 0 };
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = file_size;
    module_info.pCode = (uint32_t*)code;

    VkShaderModule shader_module;
    vkCreateShaderModule(ctx->device, &module_info, NULL, &shader_module);
    free(code);

    VkComputePipelineCreateInfo pipeline_info = { 0 };
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;

    VkPipeline pipeline;
    vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
    vkDestroyShaderModule(ctx->device, shader_module, NULL);

    return pipeline;
}