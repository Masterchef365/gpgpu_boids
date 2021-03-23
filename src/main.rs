use anyhow::Result;
use erupt::extensions::{
    khr_surface::{self, SurfaceKHR},
    khr_swapchain::{self, SwapchainKHR},
};
use erupt::vk;
use klystron::{
    mem_objects::MemObject,
    windowed::{self, hardware::SurfaceInfo},
    ApplicationInfo, Core, HardwareSelection, Memory, SharedCore, VulkanSetup,
};
use wibaeowibtnr as klystron;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{self, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let engine = Engine::new(event_loop)?;

        Ok(Self { engine })
    }

    pub fn event(&mut self, event: WindowEvent) -> Result<()> {
        Ok(())
    }

    pub fn draw(&mut self) -> Result<()> {
        Ok(())
    }
}

const FRAMES_IN_FLIGHT: u32 = 2;
struct Engine {
    window: Window,
    hardware: HardwareSelection,

    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,

    descriptor_pool: vk::DescriptorPool,
    scene_desc_set: vk::DescriptorSet,
    boid_desc_set: vk::DescriptorSet,

    //scene_data: MemObject<vk::Buffer>,

    //boid_buf_a: MemObject<vk::Buffer>,
    //boid_buf_b: MemObject<vk::Buffer>,
    //boid_buf_select: bool,

    swapchain: SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    surface_info: SurfaceInfo,
    surface: SurfaceKHR,

    //image_available: vk::Semaphore,
    //render_finished: vk::Semaphore,

    core: SharedCore,
}

impl Engine {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
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

        let (swapchain, swapchain_images) =
            build_swapchain(&core, &hardware, surface, &surface_info)?;

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

        let boids_descriptor_set_layout = unsafe {
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
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
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
            .descriptor_count(1)
        ];
        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(FRAMES_IN_FLIGHT * 2);
        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Sets
        let descriptor_set_layouts = [boids_descriptor_set_layout, scene_descriptor_set_layout];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets =
            unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?;

        let boid_desc_set = descriptor_sets[0];
        let scene_desc_set = descriptor_sets[1];

        Ok(Self {
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
            swapchain_images,
        })
    }
}

fn build_swapchain(
    core: &Core,
    hardware: &HardwareSelection,
    surface: SurfaceKHR,
    surface_info: &SurfaceInfo,
) -> Result<(SwapchainKHR, Vec<vk::Image>)> {
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
    let swapchain_images =
        unsafe { core.device.get_swapchain_images_khr(swapchain, None) }.result()?;

    Ok((swapchain, swapchain_images))
}
