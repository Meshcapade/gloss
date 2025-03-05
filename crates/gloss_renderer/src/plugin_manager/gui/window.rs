use gloss_utils::abi_stable_aliases::std_types::{RString, RVec};
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

use super::widgets::Widgets;

//similar to
// https://docs.rs/egui/latest/egui/struct.Align2.html#associatedconstant.LEFT_TOP
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub enum WindowPivot {
    LeftBottom,
    LeftCenter,
    LeftTop,
    CenterBottom,
    CenterCenter,
    CenterTop,
    RightBottom,
    RightCenter,
    RightTop,
}
//Position is normalized between [0,1]
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct WindowPosition(pub [f32; 2]);

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub enum WindowPositionType {
    Initial,
    Fixed,
}

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub enum GuiWindowType {
    Sidebar,
    FloatWindow(WindowPivot, WindowPosition, WindowPositionType),
}

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct GuiWindow {
    pub window_name: RString,
    pub window_type: GuiWindowType,
    pub widgets: RVec<Widgets>,
}
