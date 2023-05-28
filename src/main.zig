const builtin = @import("builtin");
const std = @import("std");
const shaders = @import("shaders");

// const vk = @import("vulkan");
const vk = @import("vk.zig"); // workaround for autocomplete

const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
});

const InstanceDispatch = vk.InstanceWrapper(.{
    .destroyInstance = true,
    .enumeratePhysicalDevices = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .createDevice = true,
});

const DeviceDispatch = vk.DeviceWrapper(.{
    .destroyDevice = true,
    .getDeviceQueue = true,
    .createShaderModule = true,
    .destroyShaderModule = true,
    .createPipelineLayout = true,
    .destroyPipelineLayout = true,
    .createPipelineCache = true,
    .destroyPipelineCache = true,
    .getPipelineCacheData = true,
    .createComputePipelines = true,
    .destroyPipeline = true,
    .createCommandPool = true,
    .destroyCommandPool = true,
    .allocateCommandBuffers = true,
    .beginCommandBuffer = true,
    .endCommandBuffer = true,
    .cmdBindPipeline = true,
    .cmdDispatch = true,
    .createFence = true,
    .destroyFence = true,
    .queueSubmit = true,
    .waitForFences = true,
});

const VulkanLoader = struct {
    const Self = @This();
    const library_names = switch (builtin.os.tag) {
        .windows => &[_][]const u8{"vulkan-1.dll"},
        .ios, .macos, .tvos, .watchos => &[_][]const u8{ "libvulkan.dylib", "libvulkan.1.dylib", "libMoltenVK.dylib" },
        else => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
    };

    handle: std.DynLib,
    get_instance_proc_addr: vk.PfnGetInstanceProcAddr,
    get_device_proc_addr: vk.PfnGetDeviceProcAddr,

    fn init() !Self {
        for (library_names) |library_name| {
            if (std.DynLib.open(library_name)) |library| {
                var handle = library;
                errdefer handle.close();
                return .{
                    .handle = handle,
                    .get_instance_proc_addr = handle.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.InitializationFailed,
                    .get_device_proc_addr = handle.lookup(vk.PfnGetDeviceProcAddr, "vkGetDeviceProcAddr") orelse return error.InitializationFailed,
                };
            } else |_| {}
        }
        return error.InitializationFailed;
    }

    fn deinit(self: *Self) void {
        self.handle.close();
    }
};

const QueueFamily = struct {
    physical_device: vk.PhysicalDevice,
    index: usize,
};

const Instance = struct {
    const Self = @This();

    handle: vk.Instance,
    allocation_callbacks: ?*const vk.AllocationCallbacks,
    vki: InstanceDispatch,

    fn init(vkb: BaseDispatch, loader: VulkanLoader, allocation_callbacks: ?*const vk.AllocationCallbacks) !Self {
        const extensions: []const [*:0]const u8 = &.{
            vk.extension_info.khr_portability_enumeration.name,
        };
        const instance = try vkb.createInstance(&.{
            .enabled_extension_count = extensions.len,
            .pp_enabled_extension_names = extensions.ptr,
            .flags = vk.InstanceCreateFlags{
                .enumerate_portability_bit_khr = true,
            },
        }, allocation_callbacks);
        if (InstanceDispatch.load(instance, loader.get_instance_proc_addr)) |vki| {
            errdefer vki.destroyInstance(instance, allocation_callbacks);
            return .{
                .handle = instance,
                .allocation_callbacks = allocation_callbacks,
                .vki = vki,
            };
        } else |_| {
            // use the loader to destroy instance, because we failed to load the instance dispatch
            var loaderr = loader;
            const destroy_instance = loaderr.handle.lookup(vk.PfnDestroyInstance, "vkDestroyInstance") orelse return error.InitializationFailed;
            destroy_instance(instance, allocation_callbacks);
            return error.InitializationFailed;
        }
    }

    fn deinit(self: Self) void {
        self.vki.destroyInstance(self.handle, self.allocation_callbacks);
    }

    fn enumerate_physical_devices(self: Self, allocator: std.mem.Allocator) ![]vk.PhysicalDevice {
        var count: u32 = undefined;
        _ = try self.vki.enumeratePhysicalDevices(self.handle, &count, null);
        const physical_devices = try allocator.alloc(vk.PhysicalDevice, count);
        errdefer allocator.free(physical_devices);
        _ = try self.vki.enumeratePhysicalDevices(self.handle, &count, physical_devices.ptr);
        return physical_devices;
    }

    fn select_queue_family(self: Self, physical_devices: []vk.PhysicalDevice, allocator: std.mem.Allocator, selector: *const fn (queue_family_properties: vk.QueueFamilyProperties) bool) !?QueueFamily {
        for (physical_devices) |physical_device| {
            var count: u32 = undefined;
            self.vki.getPhysicalDeviceQueueFamilyProperties(physical_device, &count, null);
            const queue_families_properties = try allocator.alloc(vk.QueueFamilyProperties, count);
            defer allocator.free(queue_families_properties);
            self.vki.getPhysicalDeviceQueueFamilyProperties(physical_device, &count, queue_families_properties.ptr);

            const context = {};
            std.sort.sort(vk.QueueFamilyProperties, queue_families_properties, context, struct {
                fn lambda(ctx: @TypeOf(context), lhs: vk.QueueFamilyProperties, rhs: vk.QueueFamilyProperties) bool {
                    _ = ctx;
                    return lhs.queue_count < rhs.queue_count or @popCount(lhs.queue_flags.toInt()) < @popCount(rhs.queue_flags.toInt());
                }
            }.lambda);

            for (queue_families_properties, 0..) |queue_family_properties, queue_family_index| {
                if (selector(queue_family_properties)) {
                    return .{
                        .physical_device = physical_device,
                        .index = queue_family_index,
                    };
                }
            }
        }
        return null;
    }
};

const Device = struct {
    const Self = @This();

    allocation_callbacks: ?*const vk.AllocationCallbacks,
    handle: vk.Device,
    vkd: DeviceDispatch,

    fn init(vki: InstanceDispatch, loader: VulkanLoader, queue_family: QueueFamily, allocation_callbacks: ?*const vk.AllocationCallbacks) ![]const Queue {
        const queue_priorities: []const f32 = &.{
            @as(f32, 1.0),
        };

        const queue_create_infos: []const vk.DeviceQueueCreateInfo = &.{
            .{
                .queue_family_index = @intCast(u32, queue_family.index),
                .queue_count = @intCast(u32, queue_priorities.len),
                .p_queue_priorities = queue_priorities.ptr,
            },
        };

        const handle = try vki.createDevice(queue_family.physical_device, &.{
            .queue_create_info_count = @intCast(u32, queue_create_infos.len),
            .p_queue_create_infos = queue_create_infos.ptr,
        }, allocation_callbacks);

        if (DeviceDispatch.load(handle, loader.get_device_proc_addr)) |vkd| {
            errdefer vkd.destroyDevice(handle, allocation_callbacks);
            const device = Device{
                .handle = handle,
                .allocation_callbacks = allocation_callbacks,
                .vkd = vkd,
            };
            return &.{Queue.init(device, queue_family.index, 0)};
        } else |_| {
            // use the loader to destroy device, because we failed to load the device dispatch
            var loaderr = loader;
            const destroy_device = loaderr.handle.lookup(vk.PfnDestroyDevice, "vkDestroyDevice") orelse return error.InitializationFailed;
            destroy_device(handle, allocation_callbacks);
            return error.InitializationFailed;
        }

        return;
    }

    fn deinit(self: Self) void {
        self.vkd.destroyDevice(self.handle, self.allocation_callbacks);
    }
};

const Queue = struct {
    const Self = @This();

    handle: vk.Queue,
    device: Device,

    fn init(device: Device, family_index: usize, index: usize) Self {
        return .{
            .device = device,
            .handle = device.vkd.getDeviceQueue(
                device.handle,
                @intCast(u32, family_index),
                @intCast(u32, index),
            ),
        };
    }
};

const ShaderModule = struct {
    const Self = @This();

    handle: vk.ShaderModule,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,

    fn init(device: Device, code: []const u8, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        return .{
            .handle = try device.vkd.createShaderModule(device.handle, &.{
                .code_size = code.len,
                .p_code = @ptrCast([*]const u32, @alignCast(4, code.ptr)),
            }, allocation_callbacks),
            .device = device,
            .allocation_callbacks = allocation_callbacks,
        };
    }

    fn deinit(self: Self) void {
        self.device.vkd.destroyShaderModule(self.device.handle, self.handle, self.allocation_callbacks);
    }
};

const PipelineLayout = struct {
    const Self = @This();

    handle: vk.PipelineLayout,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,

    fn init(device: Device, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        const descripter_set_layouts: []const vk.DescriptorSetLayout = &.{};
        return .{
            .handle = try device.vkd.createPipelineLayout(device.handle, &.{
                .p_set_layouts = descripter_set_layouts.ptr,
                .set_layout_count = descripter_set_layouts.len,
            }, allocation_callbacks),
            .device = device,
            .allocation_callbacks = allocation_callbacks,
        };
    }

    fn deinit(self: Self) void {
        self.device.vkd.destroyPipelineLayout(self.device.handle, self.handle, self.allocation_callbacks);
    }
};

const PipelineCache = struct {
    const Self = @This();

    handle: vk.PipelineCache,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,

    fn init(device: Device, data: []u8, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        return .{
            .handle = try device.vkd.createPipelineCache(device.handle, &.{
                .initial_data_size = data.len,
                .p_initial_data = data.ptr,
            }, allocation_callbacks),
            .device = device,
            .allocation_callbacks = allocation_callbacks,
        };
    }

    fn deinit(self: Self) void {
        self.device.vkd.destroyPipelineCache(self.device.handle, self.handle, self.allocation_callbacks);
    }

    fn get_data(self: Self, allocator: std.mem.Allocator) ![]u8 {
        var size: usize = undefined;
        _ = try self.device.vkd.getPipelineCacheData(self.device.handle, self.handle, &size, null);
        const data = try allocator.alloc(u8, size);
        errdefer allocator.free(data);
        _ = try self.device.vkd.getPipelineCacheData(self.device.handle, self.handle, &size, data.ptr);
        return data;
    }

    fn save(self: Self, sub_path: []const u8, allocator: std.mem.Allocator) !void {
        const file = try std.fs.cwd().createFile(sub_path, .{});
        defer file.close();
        const data = try self.get_data(allocator);
        defer allocator.free(data);
        try file.writeAll(data);
    }
};

const ComputePipelines = struct {
    const Self = @This();

    handles: []vk.Pipeline,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,
    allocator: std.mem.Allocator,

    fn init(device: Device, infos: []const vk.ComputePipelineCreateInfo, pipeline_cache: PipelineCache, allocator: std.mem.Allocator, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        const handles = try allocator.alloc(vk.Pipeline, infos.len);
        errdefer allocator.free(handles);

        _ = try device.vkd.createComputePipelines(
            device.handle,
            pipeline_cache.handle,
            @intCast(u32, infos.len),
            infos.ptr,
            allocation_callbacks,
            handles.ptr,
        );

        return .{
            .handles = handles,
            .device = device,
            .allocation_callbacks = allocation_callbacks,
            .allocator = allocator,
        };
    }

    fn deinit(self: Self) void {
        for (self.handles) |handle| {
            self.device.vkd.destroyPipeline(self.device.handle, handle, self.allocation_callbacks);
        }
        self.allocator.free(self.handles);
    }
};

const CommandPool = struct {
    const Self = @This();

    handle: vk.CommandPool,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,

    fn init(device: Device, queue_family_index: usize, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        return .{
            .handle = try device.vkd.createCommandPool(device.handle, &.{
                .queue_family_index = @intCast(u32, queue_family_index),
            }, allocation_callbacks),
            .device = device,
            .allocation_callbacks = allocation_callbacks,
        };
    }

    fn deinit(self: Self) void {
        self.device.vkd.destroyCommandPool(self.device.handle, self.handle, self.allocation_callbacks);
    }
};

const CommandBuffers = struct {
    const Self = @This();

    handles: []vk.CommandBuffer,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,
    allocator: std.mem.Allocator,

    fn init(device: Device, command_pool: CommandPool, count: u32, allocator: std.mem.Allocator, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        const handles = try allocator.alloc(vk.CommandBuffer, count);
        errdefer allocator.free(handles);

        _ = try device.vkd.allocateCommandBuffers(
            device.handle,
            &.{
                .command_pool = command_pool.handle,
                .level = vk.CommandBufferLevel.primary,
                .command_buffer_count = count,
            },
            handles.ptr,
        );

        return .{
            .handles = handles,
            .device = device,
            .allocation_callbacks = allocation_callbacks,
            .allocator = allocator,
        };
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.handles);
    }
};

const Fence = struct {
    const Self = @This();

    handle: vk.Fence,
    device: Device,
    allocation_callbacks: ?*vk.AllocationCallbacks,

    fn init(device: Device, queue_family_index: usize, allocation_callbacks: ?*vk.AllocationCallbacks) !Self {
        _ = queue_family_index;
        return .{
            .handle = try device.vkd.createFence(device.handle, &.{}, allocation_callbacks),
            .device = device,
            .allocation_callbacks = allocation_callbacks,
        };
    }

    fn deinit(self: Self) void {
        self.device.vkd.destroyFence(self.device.handle, self.handle, self.allocation_callbacks);
    }
};

fn read_file_contents(sub_path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.fs.cwd().openFile(sub_path, .{});
    defer file.close();
    const metadata = try file.metadata();
    if (metadata.kind() == .File) {
        return file.readToEndAlloc(allocator, std.math.maxInt(usize));
    } else {
        return error.WrongFileType;
    }
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{
        .verbose_log = true,
    }){};
    defer std.debug.assert(general_purpose_allocator.deinit() == .ok);
    const allocator = general_purpose_allocator.allocator();

    var loader = try VulkanLoader.init();
    defer loader.deinit();

    const vkb = try BaseDispatch.load(loader.get_instance_proc_addr);

    const instance = try Instance.init(vkb, loader, null);
    defer instance.deinit();

    var stack_fallback = std.heap.stackFallback(4 * @sizeOf(vk.PhysicalDevice) + 8 * @sizeOf(vk.QueueFamilyProperties), allocator);
    const stack_fallback_allocator = stack_fallback.get();

    const physical_devices = try instance.enumerate_physical_devices(stack_fallback_allocator);
    defer stack_fallback_allocator.free(physical_devices);

    const queue_family = try instance.select_queue_family(physical_devices, stack_fallback_allocator, struct {
        fn lambda(queue_family_properties: vk.QueueFamilyProperties) bool {
            return queue_family_properties.queue_flags.compute_bit;
        }
    }.lambda) orelse return error.NoSuitableQueueFamily;

    const queues = try Device.init(instance.vki, loader, queue_family, null);
    const queue = queues[0];
    const device = queue.device;
    defer device.deinit();

    const shader_module = try ShaderModule.init(device, &shaders.minimal_shader, null);
    defer shader_module.deinit();

    const pipeline_layout = try PipelineLayout.init(device, null);
    defer pipeline_layout.deinit();

    var data: []u8 = &.{};

    if (read_file_contents("pipeline_cache.bin", allocator)) |contents| {
        data = contents;
    } else |_| {}

    defer if (data.len > 0) allocator.free(data);

    const pipeline_cache = try PipelineCache.init(device, data, null);
    defer {
        pipeline_cache.save("pipeline_cache.bin", allocator) catch {};
        pipeline_cache.deinit();
    }

    const compute_pipelines = try ComputePipelines.init(device, &.{
        .{
            .layout = pipeline_layout.handle,
            .stage = .{
                .module = shader_module.handle,
                .stage = .{
                    .compute_bit = true,
                },
                .p_name = "main",
            },
            .base_pipeline_index = 0,
        },
    }, pipeline_cache, allocator, null);
    defer compute_pipelines.deinit();

    const command_pool = try CommandPool.init(device, queue_family.index, null);
    defer command_pool.deinit();

    const command_buffers = try CommandBuffers.init(device, command_pool, 1, allocator, null);
    defer command_buffers.deinit();

    const command_buffer = command_buffers.handles[0];
    const pipeline = compute_pipelines.handles[0];

    try device.vkd.beginCommandBuffer(command_buffer, &.{
        .flags = vk.CommandBufferUsageFlags{
            .one_time_submit_bit = true,
        },
    });
    device.vkd.cmdBindPipeline(command_buffer, vk.PipelineBindPoint.compute, pipeline);
    device.vkd.cmdDispatch(command_buffer, 1024, 1, 1);
    try device.vkd.endCommandBuffer(command_buffer);

    const fence = try Fence.init(device, queue_family.index, null);
    defer fence.deinit();

    const stage_masks: []const vk.PipelineStageFlags = &.{};

    const submits: []const vk.SubmitInfo = &.{
        .{
            .command_buffer_count = @intCast(u32, command_buffers.handles.len),
            .p_command_buffers = command_buffers.handles.ptr,
            .p_wait_dst_stage_mask = stage_masks.ptr,
        },
    };

    try device.vkd.queueSubmit(queue.handle, @intCast(u32, submits.len), submits.ptr, fence.handle);
    const fences: []const vk.Fence = &.{
        fence.handle,
    };
    const t1 = std.time.nanoTimestamp();
    _ = try device.vkd.waitForFences(device.handle, @intCast(u32, fences.len), fences.ptr, 0, std.math.maxInt(u64));
    const t2 = std.time.nanoTimestamp();
    std.debug.print("GPU took {} ns\n", .{t2 - t1});
}
