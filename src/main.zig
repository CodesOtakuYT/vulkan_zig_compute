const builtin = @import("builtin");
const std = @import("std");

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
}
