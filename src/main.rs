use anyhow::{Context, Result};
use erupt::extensions::{
    khr_surface::{self, SurfaceKHR},
    khr_swapchain::{self, SwapchainKHR},
};
use erupt::utils::decode_spv;
use erupt::vk;
use gpu_alloc::UsageFlags;
use gpu_alloc_erupt::EruptMemoryDevice;
use klystron::{
    mem_objects::MemObject,
    windowed::{self, hardware::SurfaceInfo},
    ApplicationInfo, Core, HardwareSelection, Memory, SharedCore, VulkanSetup,
};
use std::ffi::CString;
use wibaeowibtnr as klystron;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{self, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
mod camera;
use camera::MouseArcBall;

fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    let mut app = App::new(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent { event, .. } => {
                app.event(event).unwrap();
            }
            Event::MainEventsCleared => {
                app.draw().unwrap();
            }
            _ => (),
        }
    });
}

struct App {
    engine: Engine,
    camera: MouseArcBall,
    cam_matrix: [f32; 32],
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let workgroups = 22;
        let engine = Engine::new(event_loop, false, workgroups)?;
        let camera = MouseArcBall::new(Default::default(), 0.001, 0.004);

        Ok(Self {
            engine,
            camera,
            cam_matrix: [0.; 32],
        })
    }

    pub fn event(&mut self, event: WindowEvent) -> Result<()> {
        self.camera.handle_events(&event);
        let extent = self.engine.swapchain.extent;
        let mut data = [0.0; 32];
        data.iter_mut()
            .zip(
                self.camera
                    .inner
                    .matrix(extent.width, extent.height)
                    .as_slice()
                    .iter(),
            )
            .for_each(|(o, i)| *o = *i);
        self.cam_matrix = data;

        Ok(())
    }

    pub fn draw(&mut self) -> Result<()> {
        let extent = self.engine.swapchain.extent;
        self.engine.draw(self.cam_matrix)
    }
}

const COLOR_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const BOID_LOCAL_X: u32 = 16;
const BOID_SIZE: u32 = 4 * (3 + 1 + 3 + 1);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SceneData {
    pub cameras: [f32; 32],
    pub frame: u32,
}

unsafe impl bytemuck::Pod for SceneData {}
unsafe impl bytemuck::Zeroable for SceneData {}

struct Engine {
    window: Window,
    hardware: HardwareSelection,

    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,

    descriptor_pool: vk::DescriptorPool,
    scene_desc_set: vk::DescriptorSet,
    boid_desc_set: vk::DescriptorSet,

    scene_pipeline_layout: vk::PipelineLayout,
    scene_pipeline: vk::Pipeline,

    boid_pipeline_layout: vk::PipelineLayout,
    boid_pipeline: vk::Pipeline,
    boid_init_pipeline: vk::Pipeline,

    surface_info: SurfaceInfo,
    surface: SurfaceKHR,
    swapchain: Swapchain,

    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,

    render_pass: vk::RenderPass,

    scene_data: MemObject<vk::Buffer>,

    boid_buf_a: MemObject<vk::Buffer>,
    boid_buf_b: MemObject<vk::Buffer>,
    boid_buf_select: bool,

    workgroups: u32,
    vr: bool,

    frame: u32,

    core: SharedCore,
}

impl Engine {
    pub fn new(event_loop: &EventLoop<()>, vr: bool, workgroups: u32) -> Result<Self> {
        // Windowing
        let window = WindowBuilder::new()
            .with_resizable(true)
            .build(&event_loop)?;

        // Instance setup
        let mut setup = VulkanSetup::validation(vk::make_version(1, 0, 0));

        let app_info = ApplicationInfo {
            name: "Annoyta".into(),
            version: vk::make_version(1, 0, 0),
        };

        let (surface, hardware, surface_info, core) =
            windowed::basics(&app_info, &mut setup, &window)?;

        // Create command pool
        let create_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(hardware.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool =
            unsafe { core.device.create_command_pool(&create_info, None, None) }.result()?;

        // Create command buffers
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer =
            unsafe { core.device.allocate_command_buffers(&allocate_info) }.result()?[0];

        // Boid descriptors
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let boid_descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&create_info, None, None)
        }
        .result()?;

        // Scene descriptors
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let scene_descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&create_info, None, None)
        }
        .result()?;

        // Pool
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Sets
        let descriptor_set_layouts = [boid_descriptor_set_layout, scene_descriptor_set_layout];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets =
            unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?;

        let boid_desc_set = descriptor_sets[0];
        let scene_desc_set = descriptor_sets[1];

        // Whether or not the frame at this index is available (gpu-only)
        let create_info = vk::SemaphoreCreateInfoBuilder::new();
        let image_available =
            unsafe { core.device.create_semaphore(&create_info, None, None) }.result()?;

        // Whether or not the frame at this index is finished rendering (gpu-only)
        let render_finished =
            unsafe { core.device.create_semaphore(&create_info, None, None) }.result()?;

        // Load shader source
        let load_shader_module = |name: &str| -> Result<vk::ShaderModule> {
            let shader_spirv = std::fs::read(name)
                .with_context(|| format!("Shader \"{}\" failed to load", name))?;
            let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
            let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
            Ok(unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?)
        };

        let boid_module = load_shader_module("./shaders/boids.comp.spv")?;
        let boid_init_module = load_shader_module("./shaders/boids_init.comp.spv")?;
        let vertex_module = load_shader_module("./shaders/unlit.vert.spv")?;
        let fragment_module = load_shader_module("./shaders/unlit.frag.spv")?;

        // Build boid pipeline
        let main_entry_point = CString::new("main")?;
        let descriptor_set_layouts = [boid_descriptor_set_layout];
        let create_info =
            vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&descriptor_set_layouts);
        let boid_pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(boid_module)
            .name(&main_entry_point)
            .build();
        let boid_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(stage)
            .layout(boid_pipeline_layout);

        // Boid init pipeline
        let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(boid_init_module)
            .name(&main_entry_point)
            .build();
        let boid_init_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(stage)
            .layout(boid_pipeline_layout);

        // Make pipelines
        let boid_pipelines = unsafe {
            core.device.create_compute_pipelines(
                None,
                &[boid_create_info, boid_init_create_info],
                None,
            )
        }
        .result()?;
        let boid_pipeline = boid_pipelines[0];
        let boid_init_pipeline = boid_pipelines[1];

        // Create render pass
        let render_pass = create_render_pass(&core, vr)?;

        // Create swapchain
        let swapchain = Swapchain::new(
            core.clone(),
            &hardware,
            surface,
            &surface_info,
            render_pass,
            vr,
        )?;

        // Build scene pipeline
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::LINE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewport_count(1)
            .scissor_count(1);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dynamic_states);

        let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_clamp_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlagBits::_1);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];
        let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let shader_stages = [
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::VERTEX)
                .module(vertex_module)
                .name(&main_entry_point),
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::FRAGMENT)
                .module(fragment_module)
                .name(&main_entry_point),
        ];

        let descriptor_set_layouts = [scene_descriptor_set_layout];

        let create_info =
            vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&descriptor_set_layouts);

        let scene_pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS) // TODO: Play with this! For fun!
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

        let scene_create_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .input_assembly_state(&input_assembly)
            .vertex_input_state(&vertex_input)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .layout(scene_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let scene_pipeline = unsafe {
            core.device
                .create_graphics_pipelines(None, &[scene_create_info], None)
        }
        .result()?[0];

        // Destroy unused modules
        unsafe {
            for &m in &[
                boid_module,
                boid_init_module,
                vertex_module,
                fragment_module,
            ] {
                core.device.destroy_shader_module(Some(m), None);
            }
        }

        // Create boid buffers
        let n_boids = workgroups * BOID_LOCAL_X;
        let boid_storage_size = BOID_SIZE * n_boids;

        let buffer_create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(boid_storage_size as _)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let boid_buf_a = MemObject::<vk::Buffer>::new(
            &core,
            buffer_create_info,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;
        let boid_buf_b = MemObject::<vk::Buffer>::new(
            &core,
            buffer_create_info,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        let buffer_create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .size(std::mem::size_of::<SceneData>() as _)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Create scene data buffer
        let scene_data = MemObject::<vk::Buffer>::new(
            &core,
            buffer_create_info,
            UsageFlags::UPLOAD | UsageFlags::HOST_ACCESS,
        )?;

        let mut instance = Self {
            core,
            window,
            surface,
            hardware,
            surface_info,
            command_pool,
            command_buffer,
            descriptor_pool,
            boid_desc_set,
            scene_desc_set,
            swapchain,
            image_available,
            render_finished,
            boid_pipeline,
            boid_init_pipeline,
            scene_pipeline,
            workgroups,
            boid_buf_a,
            boid_buf_b,
            scene_data,
            scene_pipeline_layout,
            boid_pipeline_layout,
            boid_buf_select: false,
            render_pass,
            frame: 0,
            vr,
        };

        instance.reset_boids()?;

        Ok(instance)
    }

    fn write_boid_desc_set(&self) -> Result<()> {
        let (read_buf, write_buf) = match self.boid_buf_select {
            true => (self.boid_buf_a.instance, self.boid_buf_b.instance),
            false => (self.boid_buf_b.instance, self.boid_buf_a.instance),
        };

        unsafe {
            self.core.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(self.boid_desc_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(read_buf)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(self.boid_desc_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(write_buf)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                ],
                &[],
            );
        }
        Ok(())
    }

    fn write_scene_desc_set(&self) -> Result<()> {
        let read_buf = match self.boid_buf_select {
            true => self.boid_buf_a.instance,
            false => self.boid_buf_b.instance,
        };

        unsafe {
            self.core.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(self.scene_desc_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(self.scene_data.instance)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(self.scene_desc_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(read_buf)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                ],
                &[],
            );
        }
        Ok(())
    }

    pub fn reset_boids(&mut self) -> Result<()> {
        unsafe {
            self.core
                .device
                .queue_wait_idle(self.core.graphics_queue)
                .result()?;
            self.write_boid_desc_set()?;
            self.core
                .device
                .reset_command_buffer(self.command_buffer, None)
                .result()?;
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self.core
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .result()?;

            self.core.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.boid_pipeline_layout,
                0,
                &[self.boid_desc_set],
                &[],
            );

            self.core.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.boid_init_pipeline,
            );

            self.core
                .device
                .cmd_dispatch(self.command_buffer, self.workgroups, 1, 1);

            self.core
                .device
                .end_command_buffer(self.command_buffer)
                .result()?;
            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self.core
                .device
                .queue_submit(self.core.graphics_queue, &[submit_info], None)
                .result()?;
            self.core
                .device
                .queue_wait_idle(self.core.graphics_queue)
                .result()?;
        }
        Ok(())
    }

    pub fn draw(&mut self, camera_matrices: [f32; 32]) -> Result<()> {
        // TODO: Real sync!
        unsafe {
            self.core
                .device
                .queue_wait_idle(self.core.graphics_queue)
                .result()?;
            }

        let data = SceneData {
            cameras: camera_matrices,
            frame: self.frame,
        };
        unsafe {
            let device = EruptMemoryDevice::wrap(&self.core.device);
            self.scene_data.memory_mut().write_bytes(device, 0, bytemuck::cast_slice(&[data]))?;
        }


        // Get the index of the next swapchain image,
        // and set up a semaphore to be notified when it is ready.
        let swap_image_index = unsafe {
            self.core.device.acquire_next_image_khr(
                self.swapchain.swapchain,
                u64::MAX,
                Some(self.image_available),
                None,
                None,
            )
        };

        // Early return and invalidate swapchain
        let image_index = if swap_image_index.raw == vk::Result::ERROR_OUT_OF_DATE_KHR {
            self.swapchain = Swapchain::new(
                self.core.clone(),
                &self.hardware,
                self.surface,
                &self.surface_info,
                self.render_pass,
                self.vr,
            )?;
            return Ok(());
        } else {
            swap_image_index.unwrap() as usize
        };

        let swapchain_image = self.swapchain.images[image_index];
        let framebuffer = self.swapchain.framebuffers[image_index];

        // Update descriptor set to include the buffer
        self.write_boid_desc_set()?;
        self.write_scene_desc_set()?;

        unsafe {
            // Reset and begin
            self.core
                .device
                .reset_command_buffer(self.command_buffer, None)
                .result()?;
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self.core
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .result()?;

            // ############################# BOIDS #############################

            self.core.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.boid_pipeline_layout,
                0,
                &[self.boid_desc_set],
                &[],
            );

            // Radiation pipeline
            self.core.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.boid_pipeline,
            );

            self.core
                .device
                .cmd_dispatch(self.command_buffer, BOID_LOCAL_X, 1, 1);

            // ############################# GRAPHICS #############################

            // Set render pass
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let begin_info = vk::RenderPassBeginInfoBuilder::new()
                .framebuffer(framebuffer)
                .render_pass(self.render_pass)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .clear_values(&clear_values);

            self.core.device.cmd_begin_render_pass(
                self.command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            );

            // Scissor
            let viewports = [vk::ViewportBuilder::new()
                .x(0.0)
                .y(0.0)
                .width(self.swapchain.extent.width as f32)
                .height(self.swapchain.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)];

            let scissors = [vk::Rect2DBuilder::new()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(self.swapchain.extent)];

            self.core
                .device
                .cmd_set_viewport(self.command_buffer, 0, &viewports);

            self.core
                .device
                .cmd_set_scissor(self.command_buffer, 0, &scissors);

            // Graphics pipeline
            self.core.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.scene_pipeline_layout,
                0,
                &[self.scene_desc_set],
                &[],
            );

            self.core.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.scene_pipeline,
            );

            let n_boids = self.workgroups * BOID_LOCAL_X;
            self.core
                .device
                .cmd_draw(self.command_buffer, n_boids * 2, 1, 0, 0);

            self.core.device.cmd_end_render_pass(self.command_buffer);

            // End and submit
            self.core
                .device
                .end_command_buffer(self.command_buffer)
                .result()?;
            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self.core
                .device
                .queue_submit(self.core.graphics_queue, &[submit_info], None)
                .result()?;

            // Present
            let wait_semaphores = [self.image_available];
            let command_buffers = [self.command_buffer];
            let signal_semaphores = [self.render_finished];
            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_GRAPHICS])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            self.core
                .device
                .queue_submit(self.core.graphics_queue, &[submit_info], None)
                .result()?;

            let swapchains = [self.swapchain.swapchain];
            let image_indices = [image_index as u32];
            let present_info = khr_swapchain::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            self.core
                .device
                .queue_present_khr(self.core.graphics_queue, &present_info)
                .result()?;
        }

        self.frame += 1;

        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        todo!("Dealloc")
    }
}

fn create_render_pass(core: &Core, vr: bool) -> Result<vk::RenderPass> {
    let device = &core.device;

    // Render pass
    let color_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(COLOR_FORMAT)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(if vr {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::PRESENT_SRC_KHR
        });

    let depth_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(DEPTH_FORMAT)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let attachments = [color_attachment, depth_attachment];

    let color_attachment_refs = [vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let depth_attachment_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();

    let subpasses = [vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)];

    let dependencies = [vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let mut create_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let views = if vr { 2 } else { 1 };
    let view_mask = [!(!0 << views)];
    let mut multiview = vk::RenderPassMultiviewCreateInfoBuilder::new()
        .view_masks(&view_mask)
        .correlation_masks(&view_mask)
        .build();

    create_info.p_next = &mut multiview as *mut _ as _;

    Ok(unsafe { device.create_render_pass(&create_info, None, None) }.result()?)
}

struct Swapchain {
    swapchain: SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    depth_image: MemObject<vk::Image>,
    depth_image_view: vk::ImageView,
    framebuffers: Vec<vk::Framebuffer>,
    extent: vk::Extent2D,
    core: SharedCore,
}

impl Swapchain {
    pub fn new(
        core: SharedCore,
        hardware: &HardwareSelection,
        surface: SurfaceKHR,
        surface_info: &SurfaceInfo,
        render_pass: vk::RenderPass,
        vr: bool,
    ) -> Result<Self> {
        // Create swapchain
        let surface_caps = unsafe {
            core.instance.get_physical_device_surface_capabilities_khr(
                hardware.physical_device,
                surface,
                None,
            )
        }
        .result()?;
        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
            image_count = surface_caps.max_image_count;
        }

        let create_info = khr_swapchain::SwapchainCreateInfoKHRBuilder::new()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(windowed::COLOR_FORMAT)
            .image_color_space(windowed::COLOR_SPACE)
            .image_extent(surface_caps.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(khr_surface::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(surface_info.present_mode)
            .clipped(true)
            .old_swapchain(khr_swapchain::SwapchainKHR::null());

        let swapchain =
            unsafe { core.device.create_swapchain_khr(&create_info, None, None) }.result()?;

        // Swapchain images
        let images = unsafe { core.device.get_swapchain_images_khr(swapchain, None) }.result()?;

        let image_views = images
            .iter()
            .map(|image| -> Result<vk::ImageView> {
                let create_info = vk::ImageViewCreateInfoBuilder::new()
                    .image(*image)
                    .view_type(vk::ImageViewType::_2D)
                    .format(COLOR_FORMAT)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(if vr { 2 } else { 1 })
                            .build(),
                    );
                Ok(unsafe { core.device.create_image_view(&create_info, None, None) }.result()?)
            })
            .collect::<Result<Vec<_>>>()?;

        // Depth image
        let layers = if vr { 2 } else { 1 };
        let create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(
                vk::Extent3DBuilder::new()
                    .width(surface_caps.current_extent.width)
                    .height(surface_caps.current_extent.height)
                    .depth(1)
                    .build(),
            )
            .mip_levels(1)
            .array_layers(layers)
            .format(DEPTH_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let depth_image =
            MemObject::<vk::Image>::new(&core, create_info, UsageFlags::FAST_DEVICE_ACCESS)?;

        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(depth_image.instance)
            .view_type(vk::ImageViewType::_2D)
            .format(DEPTH_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(layers)
                    .build(),
            );
        let depth_image_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        let framebuffers = image_views
            .iter()
            .map(|&image_view| -> Result<vk::Framebuffer> {
                let attachments = [image_view, depth_image_view];
                let create_info = vk::FramebufferCreateInfoBuilder::new()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(surface_caps.current_extent.width)
                    .height(surface_caps.current_extent.height)
                    .layers(1);

                let framebuffer =
                    unsafe { core.device.create_framebuffer(&create_info, None, None) }.result()?;
                Ok(framebuffer)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            swapchain,
            images,
            extent: surface_caps.current_extent,
            image_views,
            core,
            depth_image_view,
            depth_image,
            framebuffers,
        })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device
                .destroy_swapchain_khr(Some(self.swapchain), None);
            for view in self.image_views.drain(..) {
                self.core.device.destroy_image_view(Some(view), None);
            }
        }
        self.depth_image.free(&self.core);
    }
}
