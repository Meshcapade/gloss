//! Message types for network synchronization.
//!
//! These messages represent the different types of scene changes that can be
//! transmitted over the network.

use serde::{Deserialize, Serialize};

/// Type identifier for components, used to identify which component type is being sent.
/// This allows the receiver to know how to deserialize and apply the component.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentTypeId(pub String);

impl ComponentTypeId {
    pub fn of<T: 'static>() -> Self {
        ComponentTypeId(std::any::type_name::<T>().to_string())
    }
}

/// Represents a single scene change that can be sent over the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneMessage {
    /// A component was inserted or updated on an entity
    ComponentInsert {
        entity_name: String,
        component_type: ComponentTypeId,
        /// Serialized component data (using bincode or similar)
        data: Vec<u8>,
    },

    /// An entity was spawned with a name
    EntitySpawn {
        entity_name: String,
        is_renderable: bool,
        is_on_gui: bool,
    },

    /// Batch of messages which is usually the result of `sender.start_frame()` then sending messages and then finalizing with `sender.end_frame()`
    Batch(Vec<SceneMessage>),

    /// Marks the beginning of a frame - all messages until `FrameEnd` belong together
    FrameBegin {
        /// Frame number for ordering
        frame_number: u64,
    },

    /// Marks the end of a frame
    FrameEnd {
        /// Frame number matching the `FrameBegin`
        frame_number: u64,
    },

    /// Heartbeat/ping message for connection health
    Ping { timestamp_ms: u64 },

    /// Response to ping
    Pong { timestamp_ms: u64 },
}

/// Frame header for network packets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageFrame {
    /// Sequence number for ordering
    pub sequence: u64,
    /// The actual message payload
    pub message: SceneMessage,
}

impl MessageFrame {
    pub fn new(sequence: u64, message: SceneMessage) -> Self {
        Self { sequence, message }
    }

    //sometimes I want to apply each scene message one by one so that I can apply the command buffer after each one
    //therefore this unzips the batch scene message and just gives scene messages
    pub fn to_individual_messages(&self) -> Vec<Self> {
        let msg = match &self.message {
            SceneMessage::Batch(messages) => messages.clone(),
            _ => vec![self.message.clone()],
        };
        msg.into_iter()
            .map(|message| Self {
                sequence: self.sequence,
                message,
            })
            .collect()
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map(|(result, _)| result)
    }
}
