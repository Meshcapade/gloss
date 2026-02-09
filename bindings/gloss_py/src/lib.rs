#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::needless_pass_by_value)] //a lot of py arrays are passed by value but that is actually fine
#![allow(clippy::return_self_not_must_use)]

#[cfg(feature = "burn-torch")]
use components::{colors_pytensor::PyColorsPyTensor, verts_pytensor::PyVertsPyTensor};

use components::{
    colors::PyColors,
    colors_gpu::PyColorsGPU,
    diffuse_img::PyDiffuseImg,
    diffuse_tex::PyDiffuseTex,
    edges::PyEdges,
    faces::PyFaces,
    img_config::PyImgConfig,
    model_matrix::PyModelMatrix,
    normal_img::PyNormalImg,
    normals::PyNormals,
    projection_with_fov::PyProjectionWithFov,
    tangents::PyTangents,
    transport_config::PyTransportConfig,
    uvs::PyUVs,
    verts::PyVerts,
    // verts_gpu::PyVertsGPU,
    vis_edges::{PyLineColorType, PyVisLines},
    vis_mesh::{PyMeshColorType, PyVisMesh},
    vis_outline::PyVisOutline,
    vis_points::{PyPointColorType, PyVisPoints},
};
use entity_builder::PyEntityBuilder;
use img::PyDynImage;
use plugin::PyPluginList;
use pyo3::prelude::*;
use resources::scene_receiver::PySceneReceiver;
use resources::scene_sender::PySceneSender;

pub mod actor;
pub mod builders;
pub mod camera;
pub mod components;
pub mod device;
pub mod entity_builder;
pub mod geom;
pub mod global_backend;
pub mod gpu;
pub mod img;
pub mod logger;
pub mod plugin;
pub mod queue;
pub mod resources;
pub mod scene;
pub mod scene_receiver_plugin;
#[cfg(feature = "burn-torch")]
pub mod tensor_utils;
pub mod texture;
#[cfg(feature = "burn-torch")]
pub mod torch2burn_test;
pub mod viewer;
pub mod viewer_dummy;
pub mod viewer_headless;

use actor::PyActorMut;
use camera::PyCamera;
use device::PyDevice;
use geom::{PyGeom, PyIndirRemovalPolicy, PySplatType};
pub use gpu::PyGpu;
use logger::{gloss_setup_logger, gloss_setup_logger_from_config_file, PyLogLevel, PyLogLevelCaps};
use queue::PyQueue;
use scene::PyScene;
use texture::PyTexture;
#[cfg(feature = "burn-torch")]
use torch2burn_test::PyTorch2BurnTest;
use viewer::PyViewer;
use viewer_dummy::PyViewerDummy;
use viewer_headless::PyViewerHeadless;

use crate::{builders::PyBuilders, scene_receiver_plugin::PySceneReceiverPlugin};

use crate::global_backend::init_global_burn_backend;

/// A Python module implemented in Rust using tch to manipulate PyTorch
/// objects.
#[pymodule]
#[pyo3(name = "gloss")]
#[allow(clippy::missing_errors_doc)]
pub fn extension(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create submodules
    let log_module = PyModule::new_bound(_py, "log")?;
    let backend_module = PyModule::new_bound(_py, "backend")?;
    let components_module = PyModule::new_bound(_py, "components")?;
    let types_module = PyModule::new_bound(_py, "types")?;
    let builders_module = PyModule::new_bound(_py, "builders")?;
    let network_module = PyModule::new_bound(_py, "network")?;

    // Add core classes to main module
    m.add_class::<PyTexture>()?;
    m.add_class::<PyViewer>()?;
    m.add_class::<PyViewerHeadless>()?;
    m.add_class::<PyViewerDummy>()?;
    m.add_class::<PyCamera>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyGpu>()?;
    m.add_class::<PyQueue>()?;
    m.add_class::<PyPluginList>()?;
    m.add_class::<PyActorMut>()?;
    m.add_class::<PyDynImage>()?;
    m.add_class::<PyGeom>()?;
    #[cfg(feature = "burn-torch")]
    m.add_class::<PyTorch2BurnTest>()?;

    // Initialize submodules
    add_submod_log(_py, &log_module)?;
    add_submod_backend(_py, &backend_module)?;
    add_submod_components_sm(_py, &components_module)?;
    add_submod_types(_py, &types_module)?;
    add_submod_builders(_py, &builders_module)?;
    add_submod_network(_py, &network_module)?;

    // Register submodules in sys.modules
    let sys = _py.import_bound("sys")?.getattr("modules")?;
    sys.set_item("gloss.log", log_module.as_ref())?;
    sys.set_item("gloss.backend", backend_module.as_ref())?;
    sys.set_item("gloss.components", components_module.as_ref())?;
    sys.set_item("gloss.types", types_module.as_ref())?;
    sys.set_item("gloss.builders", builders_module.as_ref())?;
    sys.set_item("gloss.network", network_module.as_ref())?;

    // Add submodules to main module
    m.add_submodule(&log_module)?;
    m.add_submodule(&components_module)?;
    m.add_submodule(&types_module)?;
    m.add_submodule(&builders_module)?;
    m.add_submodule(&network_module)?;

    Ok(())
}

#[pymodule]
fn add_submod_log(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLogLevel>()?;
    m.add_class::<PyLogLevelCaps>()?;
    m.add_function(wrap_pyfunction!(gloss_setup_logger, m)?)?;
    m.add_function(wrap_pyfunction!(gloss_setup_logger_from_config_file, m)?)?;
    Ok(())
}

#[pymodule]
fn add_submod_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_global_burn_backend, m)?)?;
    Ok(())
}

#[pymodule]
fn add_submod_components_sm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVerts>()?;
    // m.add_class::<PyVertsGPU>()?;
    m.add_class::<PyNormals>()?;
    m.add_class::<PyUVs>()?;
    m.add_class::<PyEdges>()?;
    m.add_class::<PyTangents>()?;
    m.add_class::<PyColors>()?;
    m.add_class::<PyColorsGPU>()?;
    m.add_class::<PyFaces>()?;
    m.add_class::<PyVisLines>()?;
    m.add_class::<PyVisMesh>()?;
    m.add_class::<PyVisPoints>()?;
    m.add_class::<PyVisOutline>()?;
    m.add_class::<PyModelMatrix>()?;
    m.add_class::<PyDiffuseImg>()?;
    m.add_class::<PyDiffuseTex>()?;
    m.add_class::<PyNormalImg>()?;
    m.add_class::<PyProjectionWithFov>()?;
    #[cfg(feature = "burn-torch")]
    {
        m.add_class::<PyColorsPyTensor>()?;
        m.add_class::<PyVertsPyTensor>()?;
    }
    Ok(())
}

#[pymodule]
fn add_submod_types(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeshColorType>()?;
    m.add_class::<PyPointColorType>()?;
    m.add_class::<PyLineColorType>()?;
    m.add_class::<PySplatType>()?;
    m.add_class::<PyIndirRemovalPolicy>()?;
    m.add_class::<PyImgConfig>()?;
    Ok(())
}

#[pymodule]
fn add_submod_builders(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEntityBuilder>()?;
    m.add_class::<PyBuilders>()?;
    Ok(())
}

#[pymodule]
fn add_submod_network(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransportConfig>()?;
    m.add_class::<PySceneSender>()?;
    m.add_class::<PySceneReceiver>()?;
    m.add_class::<PySceneReceiverPlugin>()?;
    Ok(())
}
