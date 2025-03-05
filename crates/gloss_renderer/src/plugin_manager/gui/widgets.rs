use gloss_utils::abi_stable_aliases::std_types::RVec;
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

use super::{button::Button, checkbox::Checkbox, selectable::SelectableList, slider::Slider};

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub enum Widgets {
    Slider(Slider),
    Checkbox(Checkbox),
    Button(Button),
    SelectableList(SelectableList),
    Horizontal(RVec<Widgets>),
}
