//! Networked Scene Synchronization for Remote Visualization
//!
//! This module enables real-time scene and component synchronization between a sender (e.g., a headless process or remote server) and a receiver (e.g., a local renderer or visualization client) over the network.
//!
//! ## Overview
//!
//! - **Sender**: Prepares and batches scene/component updates, then transmits them to a receiver. The sender can be a process running remotely (e.g., on an SSH server or in a compute cluster) and is responsible for serializing and sending CPU-side component data.
//! - **Receiver**: Listens for incoming scene/component updates, deserializes them, and applies them to its local ECS world for visualization. The receiver is typically a local or GUI-enabled process.
//!
//! The workflow is analogous to the Python example scripts:
//! - `scene_sender.py`: Sets up a sender, batches scene changes, and sends them over the network.
//! - `scene_receiver.py`: Sets up a receiver, listens for updates, and visualizes the received scene.
//!
//! ## Usage
//!
//! ### Sender Side (Example)
//! ```ignore
//! use gloss_renderer::network::{SceneSender, TransportConfig};
//! // Set up transport configuration (address/port)
//! let config = TransportConfig {
//!     address: "127.0.0.1".to_string(),
//!     port: 46377,
//!     ..Default::default()
//! };
//! let mut sender = SceneSender::new(config);
//! sender.start_listening();
//! sender.try_connect_to_receiver();
//! // ...
//! // Batch and send scene/component updates
//! sender.start_batch();
//! // sender.send_component(...)
//! sender.end_batch();
//! ```
//!
//! ### Receiver Side (Example)
//! ```ignore
//! use gloss_renderer::network::{SceneReceiver, TransportConfig, SceneReceiverPlugin};
//! // Set up transport configuration (address/port)
//! let config = TransportConfig {
//!     address: "127.0.0.1".to_string(),
//!     port: 46377,
//!     ..Default::default()
//! };
//! let mut receiver = SceneReceiver::new(config);
//! viewer.add_resource(receiver);
//! viewer.insert_plugin(SceneReceiverPlugin::new(true));
//! ```
//!
//! ## Notes
//!
//! - Only CPU-side components that implement `serde::Serialize` can be sent. GPU-side data must be recreated on the receiver.
//! - Batching updates (between `start_batch` and `end_batch`) ensures atomic application of scene changes.
//! - The receiver can be run as a plugin or resource in a viewer or ECS system.
//! - The sender and receiver must agree on the set of serializable components and their registration.
//!
//! See the Python scripts `scene_sender.py` and `scene_receiver.py` for a practical workflow analogy.

pub mod component_registry;
pub mod messages;
pub mod receiver;
pub mod receiver_plugin;
pub mod sender;
pub mod serializable_components;
pub mod transport;

pub use component_registry::*;
pub use messages::*;
pub use receiver::*;
pub use receiver_plugin::*;
pub use sender::*;
use serde::{de::DeserializeOwned, Serialize};
pub use serializable_components::*;
pub use transport::*;

// Trait for checking if components can be sent over the network.
/// This trait must be implemented for all components that want network functionality.
/// Usually this is implemented for Serializable objects like `SerializableVerts` and not for Verts themselves
/// You can usually implement both `NetworkSendable` and  `NetworkReceivable ` using the macro  `impl_network_sendable_and_receivable `
pub trait NetworkSendable {
    /// Serialize this component to bytes for network transmission.
    #[allow(clippy::missing_errors_doc)]
    fn serialize_to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError>;

    /// Get the component type identifier for this component type.
    fn component_type_id(&self) -> crate::network::ComponentTypeId;
}

/// Trait for components that can be received over the network.
pub trait NetworkReceivable: Serialize + DeserializeOwned + Clone {
    /// Get the component type identifier for this component type.
    fn component_type_id() -> crate::network::ComponentTypeId;
}

#[macro_export]
macro_rules! impl_network_sendable_and_receivable {
    ( $($component_type:ty),* $(,)? ) => {
        $(
            // Implement NetworkSendable for serializable components
            impl $crate::network::NetworkSendable for $component_type {
                fn serialize_to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
                    bincode::serde::encode_to_vec(self, bincode::config::standard())
                }

                fn component_type_id(&self) -> $crate::network::ComponentTypeId {
                    $crate::network::ComponentTypeId::of::<$component_type>()
                }
            }

            // Implement NetworkReceivable for serializable components
            impl $crate::network::NetworkReceivable for $component_type {
                fn component_type_id() -> $crate::network::ComponentTypeId {
                    $crate::network::ComponentTypeId::of::<$component_type>()
                }
            }
        )*
    };
}
