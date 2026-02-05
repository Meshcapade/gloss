use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::network::TransportConfig;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;

#[pyclass(name = "TransportConfig", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyTransportConfig {
    pub inner: TransportConfig,
}
#[pymethods]
impl PyTransportConfig {
    #[new]
    #[pyo3(signature = (address=None, port=None, auto_reconnect=None, reconnect_delay_ms=None, connect_timeout_ms=None, ))]
    #[pyo3(
        text_signature = "(address: Optional[str] = None, port: Optional[int] = None, auto_reconnect: Optional[bool] = None,  reconnect_delay_ms: Optional[int] = None, connect_timeout_ms: Optional[int] = None) -> TransportConfig"
    )]
    pub fn new(
        address: Option<String>,
        port: Option<u16>,
        auto_reconnect: Option<bool>,
        reconnect_delay_ms: Option<u64>,
        connect_timeout_ms: Option<u64>,
    ) -> Self {
        let def = TransportConfig::default();

        let address = address.unwrap_or(def.address);
        let port = port.unwrap_or(def.port);
        let auto_reconnect = auto_reconnect.unwrap_or(def.auto_reconnect);
        let reconnect_delay_ms = reconnect_delay_ms.unwrap_or(def.reconnect_delay_ms);
        let connect_timeout_ms = connect_timeout_ms.unwrap_or(def.connect_timeout_ms);

        PyTransportConfig {
            inner: TransportConfig {
                address,
                port,
                auto_reconnect,
                reconnect_delay_ms,
                connect_timeout_ms,
            },
        }
    }
}
