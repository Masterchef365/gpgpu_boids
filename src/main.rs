use erupt::vk;
use wibaeowibtnr as klystron;
use klystron::{
    mem_objects::MemObject, windowed, ApplicationInfo, Core, HardwareSelection, Memory, SharedCore,
    VulkanSetup,
};
use anyhow::Result;
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

        Ok(Self {
            engine,
        })
    }

    pub fn event(&mut self, event: WindowEvent) -> Result<()> {
        Ok(())
    }

    pub fn draw(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Engine {
    window: Window,
    core: SharedCore,
}

impl Engine {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let window = WindowBuilder::new()
            .with_resizable(true)
            .build(&event_loop)?;

        let mut setup = VulkanSetup::validation(vk::make_version(1, 0, 0));

        let app_info = ApplicationInfo {
            name: "Annoyta".into(),
            version: vk::make_version(1, 0, 0),
        };

        let (surface, hardware, surface_info, core) =
            windowed::basics(&app_info, &mut setup, &window)?;

        Ok(Self {
            core,
            window,
        })
    }
}
