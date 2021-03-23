use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

/// An arcball camera
pub struct ArcBall {
    pub pivot: Point3<f32>,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub clipping: (f32, f32),
}

impl ArcBall {
    /// Extract the camera matrix
    pub fn matrix(&self, width: u32, height: u32) -> Matrix4<f32> {
        let mut perspective = Matrix4::new_perspective(
            width as f32 / height as f32,
            self.fov,
            self.clipping.0,
            self.clipping.1,
        );
        perspective[(1, 1)] *= -1.;
        perspective * self.view()
    }

    /// View matrix
    pub fn view(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(
            &(self.pivot + self.eye()),
            &self.pivot,
            &Vector3::new(0.0, 1.0, 0.0),
        )
    }

    /// Eye position
    pub fn eye(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos().abs(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos().abs(),
        ) * self.distance
    }
}

impl Default for ArcBall {
    fn default() -> Self {
        Self {
            pivot: Point3::origin(),
            distance: 15.0,
            yaw: 1.0,
            pitch: 1.0,
            fov: 45.0f32.to_radians(),
            clipping: (0.1, 2000.0),
        }
    }
}

pub struct MouseArcBall {
    pub inner: ArcBall,
    pub pan_sensitivity: f32,
    pub swivel_sensitivity: f32,
    last_mouse_position: Option<(f64, f64)>,
    left_is_clicked: bool,
    right_is_clicked: bool,
}

impl MouseArcBall {
    pub fn new(inner: ArcBall, pan_sensitivity: f32, swivel_sensitivity: f32) -> Self {
        Self {
            inner,
            pan_sensitivity,
            swivel_sensitivity,
            last_mouse_position: None,
            left_is_clicked: false,
            right_is_clicked: false,
        }
    }

    pub fn handle_events(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let &PhysicalPosition { x, y } = position;
                if let Some((last_x, last_y)) = self.last_mouse_position {
                    let x_delta = (last_x - x) as f32;
                    let y_delta = (last_y - y) as f32;
                    if self.left_is_clicked {
                        self.mouse_pivot(x_delta, y_delta);
                    } else if self.right_is_clicked {
                        self.mouse_pan(x_delta, y_delta);
                    }
                }
                self.last_mouse_position = Some((x, y));
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => self.left_is_clicked = *state == ElementState::Pressed,
                MouseButton::Right => self.right_is_clicked = *state == ElementState::Pressed,
                _ => (),
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_x, y) = delta {
                    self.inner.distance += y * 0.3;
                    if self.inner.distance <= 0.01 {
                        self.inner.distance = 0.01;
                    }
                }
            }
            _ => (),
        }
    }

    fn mouse_pivot(&mut self, delta_x: f32, delta_y: f32) {
        use std::f32::consts::FRAC_PI_2;
        self.inner.yaw -= delta_x * self.swivel_sensitivity;
        self.inner.pitch -= delta_y * self.swivel_sensitivity
            .max(-FRAC_PI_2)
            .min(FRAC_PI_2);
    }

    fn mouse_pan(&mut self, delta_x: f32, delta_y: f32) {
        let delta = Vector4::new(
            (delta_x as f32) * self.inner.distance,
            (-delta_y as f32) * self.inner.distance,
            0.0,
            0.0,
        ) * self.pan_sensitivity;
        let view_inv = self.inner.view().try_inverse().unwrap();
        self.inner.pivot += (view_inv * delta).xyz();
    }
}
