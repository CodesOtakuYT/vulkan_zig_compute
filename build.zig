const std = @import("std");
const vkgen = @import("lib/vulkan-zig/generator/index.zig");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(builder: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = builder.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = builder.standardOptimizeOption(.{});

    const exe = builder.addExecutable(.{
        .name = "vulkan_zig_compute",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Allows windows to build for ReleaseSmall
    if (exe.optimize == .ReleaseSmall) exe.strip = true;

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    builder.installArtifact(exe);

    // Create a step that generates vk.zig (stored in zig-cache) from the provided vulkan registry.
    const gen = vkgen.VkGenerateStep.create(builder, "/usr/share/vulkan/registry/vk.xml");
    // Add the generated file as package to the final executable
    exe.addModule("vulkan", gen.getModule());

    const shader_comp = vkgen.ShaderCompileStep.create(
        builder,
        &[_][]const u8{ "glslc", "--target-env=vulkan1.0" }, // Path to glslc and additional parameters
        "-o",
    );
    shader_comp.add("minimal_shader", "shaders/minimal.comp", .{});
    exe.addModule("shaders", shader_comp.getModule());
    const zigimg_module = builder.createModule(.{
        .source_file = std.Build.FileSource.relative("lib/zigimg/zigimg.zig"),
    });
    exe.addModule("zigimg", zigimg_module);

    exe.linkLibC();

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = builder.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(builder.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (builder.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = builder.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const unit_tests = builder.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = builder.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = builder.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
