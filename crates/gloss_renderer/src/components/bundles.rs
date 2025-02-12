use gloss_hecs::Bundle;

use super::{CamController, LightEmit, PosLookat, Projection};

extern crate nalgebra as na;

#[derive(Bundle, Default)]
pub struct CamBundle {
    pub pos_lookat: PosLookat,
    pub projection: Projection,
    pub controller: CamController,
}

#[derive(Bundle, Default)]
pub struct SpotLightBundle {
    pub pos_lookat: PosLookat,
    pub projection: Projection,
    pub light_emit: LightEmit,
}
