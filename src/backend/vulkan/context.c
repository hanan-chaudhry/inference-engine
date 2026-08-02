#include "../../../inc/backend/vk_core.h"
#include <stdio.h>
#include <stdlib.h>

bool vk_init(VkContext* ctx) {

    // 1. Create Vulkan Instance
    VkApplicationInfo app_info = { 0 };
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Transformer Inference Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "NoEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_4;

    // debugging
    const char* validation_layers[] = { "VK_LAYER_KHRONOS_validation" };

    VkInstanceCreateInfo create_info = { 0 };
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = validation_layers;

    if (vkCreateInstance(&create_info, NULL, &ctx->instance) != VK_SUCCESS) {
        printf("ERROR: Failed to create Vulkan instance. (Tip: Check your Fedora validation layer packages)\n");
        return false;
    }

    // 2. choose physical device (GPU) that supports compute operations
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    if (device_count == 0) {
        printf("ERROR: No GPUs found with Vulkan support!\n");
        vkDestroyInstance(ctx->instance, NULL);
        return false;
    }

    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);

    // iterate and select a device that supports compute math.
    ctx->physical_device = VK_NULL_HANDLE;
    ctx->queue_family_index = UINT32_MAX;

    for (uint32_t i = 0; i < device_count; i++) {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, NULL);

        VkQueueFamilyProperties* queue_families = malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, queue_families);

        // 3. find compute queue family index
        for (uint32_t j = 0; j < queue_family_count; j++) {
            if (queue_families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx->physical_device = devices[i];
                ctx->queue_family_index = j;
                break;
            }
        }
        free(queue_families);
        if (ctx->physical_device != VK_NULL_HANDLE) break;
    }
    free(devices);

    if (ctx->physical_device == VK_NULL_HANDLE) {
        printf("ERROR: Failed to find a GPU that supports compute operations.\n");
        vkDestroyInstance(ctx->instance, NULL);
        return false;
    }

    // print the selected GPU name
    VkPhysicalDeviceProperties device_props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &device_props);
    printf("Selected GPU Device: %s\n", device_props.deviceName);

    // 4. Creating logical device (the connection to the GPU)
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = { 0 };
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = ctx->queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    // Enable maintenance4 feature for better memory management
    VkPhysicalDeviceMaintenance4Features maint4_features = { 0 };
    maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    maint4_features.maintenance4 = VK_TRUE;

    VkDeviceCreateInfo device_create_info = { 0 };
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &maint4_features;

    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    if (vkCreateDevice(ctx->physical_device, &device_create_info, NULL, &ctx->device) != VK_SUCCESS) {
        printf("ERROR: Failed to create logical device connection.\n");
        vkDestroyInstance(ctx->instance, NULL);
        return false;
    }

    // 5. get compute queue
    vkGetDeviceQueue(ctx->device, ctx->queue_family_index, 0, &ctx->compute_queue);

    // 6. create command pool
    VkCommandPoolCreateInfo pool_info = { 0 };
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx->queue_family_index;

    if (vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool) != VK_SUCCESS) {
        printf("ERROR: Failed to create command pool.\n");
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        return false;
    }

    printf("Vulkan Context Initialized successfully!\n");
    return true;
}

// cleanup in reverse order of creation

void vk_cleanup(VkContext* ctx) {
    if (!ctx) return;

    if (ctx->command_pool) {
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    }
    if (ctx->device) {
        vkDestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance) {
        vkDestroyInstance(ctx->instance, NULL);
    }
    printf("Vulkan Context gracefully destroyed.\n");
}