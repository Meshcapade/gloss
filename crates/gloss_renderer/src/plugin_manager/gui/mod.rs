pub mod button;
pub mod checkbox;
pub mod selectable;
pub mod slider;
pub mod widgets;
pub mod window;

//when we do "use crate::gui::*" make it so that we can just use directly the
// components without mentioning cam_comps for example
pub use button::*;
pub use checkbox::*;
pub use selectable::*;
pub use slider::*;
pub use widgets::*;
pub use window::*;
