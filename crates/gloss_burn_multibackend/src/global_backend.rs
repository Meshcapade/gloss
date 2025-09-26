use once_cell::sync::OnceCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalBackend {
    Candle,
    NdArray,
    Wgpu,
}

pub static GLOBAL_BURN_BACKEND: OnceCell<GlobalBackend> = OnceCell::new();

/// Check if the global backend for Burn has been initialized
pub fn has_global_burn_backend() -> bool {
    GLOBAL_BURN_BACKEND.get().is_some()
}

// #[cfg(not(feature = "external_global"))]
pub fn init_global_burn_backend(backend: GlobalBackend) {
    GLOBAL_BURN_BACKEND.set(backend).unwrap();
}

// #[cfg(not(feature = "external_global"))]
pub fn get_global_burn_backend() -> Option<GlobalBackend> {
    GLOBAL_BURN_BACKEND.get().copied()
}
