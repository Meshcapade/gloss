/// Identifies a mesh that will be rendered to the screen
pub struct Renderable;

/// Identifies a mesh that has been modified in some way and we need to update
/// the shadow map
pub struct ShadowMapDirty;

/// Marker component to specify that this entity is only visible in the GUI
/// This is mostly useful when wanting to display a window inside the GUI with a texture
/// in which case we will spawn a gui entity and add the `DiffuseImg` component
pub struct OnGui;
