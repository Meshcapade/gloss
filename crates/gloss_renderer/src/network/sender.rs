//! Scene sender functionality for networked scene synchronization.
//!
//! This module provides the `SceneSender`, which manages network transport and batching of scene/component updates for remote visualization.
//!
//! The sender is typically used in a headless or remote process to push scene/component changes to a receiver (e.g., a local renderer or visualization client).
//!
//! ## Typical Usage
//!
//! 1. **Configure the sender** with a `TransportConfig` (address/port).
//! 2. **Start listening** for receiver connections, or connect to a receiver.
//! 3. **Add the sender as a resource** to your ECS world or viewer.
//! 4. **Batch scene/component updates** between `start_batch()` and `end_batch()` (or similar methods), ensuring atomic updates.
//! 5. **Insert components/entities** as usual; the sender will serialize and transmit them to the receiver.
//!
//! ## Example
//! ```ignore
//! use gloss_renderer::network::{SceneSender, TransportConfig};
//! // Set up transport configuration (address/port)
//! let config = TransportConfig {
//!     address: "127.0.0.1".to_string(),
//!     port: 46377,
//!     ..Default::default()
//! };
//! let mut sender = SceneSender::new(config);
//! sender.start_listening().unwrap();
//! sender.try_connect_to_receiver();
//! // Add as resource to your ECS world or viewer
//! viewer.add_resource(sender);
//! // Batch and send scene/component updates
//! viewer.start_batch_net_sending();
//! // ... create or update entities/components ...
//! viewer.end_batch_net_sending();
//! ```
//!
//! ## Notes
//! - Only components implementing `NetworkSendable` and registered with the sender can be transmitted.
//! - Batching updates ensures atomic application on the receiver side.
//! - The sender can be used in both client and server modes, depending on your network topology.
//! - See the Python `scene_sender.py` for a practical workflow analogy.
use crate::network::NetworkSendable;

use super::component_registry::SendableComponentRegistry;
use super::messages::{ComponentTypeId, MessageFrame, SceneMessage};
use super::transport::{Transport, TransportConfig, TransportError};
use log::{error, trace};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct SceneSender {
    /// The underlying transport for sending messages (Arc for Clone support)
    transport: Option<Arc<dyn Transport>>,
    /// Sequence number for message ordering
    sequence: u64,
    /// Current frame number
    frame_number: u64,
    /// Transport configuration
    config: TransportConfig,
    /// Batch buffer for accumulating messages within a frame
    batch_buffer: Vec<SceneMessage>,
    /// Whether we're currently in a frame (between `begin_frame` and `end_frame`)
    in_frame: bool,
    /// Last time we attempted to connect
    last_connect_attempt: Option<Instant>,
    /// Interval between connection attempts
    connect_retry_interval: Duration,
    /// Registry for network sendable components
    component_registry: SendableComponentRegistry,
}

impl SceneSender {
    /// Create a new scene sender (not connected)
    pub fn new(config: TransportConfig) -> Self {
        Self {
            transport: None,
            sequence: 0,
            frame_number: 0,
            config,
            batch_buffer: Vec::new(),
            in_frame: false,
            last_connect_attempt: None,
            connect_retry_interval: Duration::from_millis(1000), // Try every 1 second
            component_registry: SendableComponentRegistry::default(),
        }
    }

    /// Set the transport
    pub fn set_transport(&mut self, transport: Arc<dyn Transport>) {
        self.transport = Some(transport);
    }

    /// Start listening as a TCP server without blocking.
    /// The sender will automatically accept connections when receivers try to connect.
    /// Messages will be dropped if no receiver is connected, which allows the sender to continue running.
    ///
    /// # Errors
    ///
    /// Returns `Err(TransportError)` if transport initialization or starting the listener fails; errors originate from
    /// `TcpServer::new` or `TcpServer::listen`.
    pub fn start_listening(&mut self) -> Result<(), TransportError> {
        use super::transport::TcpServer;

        let addr = format!("{}:{}", self.config.address, self.config.port);
        let tcp_server = TcpServer::new(&addr)?;
        tcp_server.listen()?;
        self.set_transport(Arc::new(tcp_server));
        Ok(())
    }

    /// Non-blocking connect attempt - returns immediately even if no receiver
    pub fn try_connect_to_receiver(&mut self) {
        if let Some(ref transport) = self.transport {
            // Only log for actual reconnection attempts, not every message
            if let Err(e) = transport.connect() {
                trace!("Connection attempt failed: {e}");
            }
        }
    }

    // /// Check if we should attempt to reconnect
    fn should_attempt_reconnect(&self) -> bool {
        if self.is_connected() {
            return false;
        }

        match self.last_connect_attempt {
            None => true, // Never attempted
            Some(last_attempt) => last_attempt.elapsed() >= self.connect_retry_interval,
        }
    }

    // /// Attempt connection if it's time to do so
    fn maybe_reconnect(&mut self) {
        if self.should_attempt_reconnect() {
            self.last_connect_attempt = Some(Instant::now());
            if let Some(ref transport) = self.transport {
                // Only log for actual reconnection attempts, not every message
                if let Err(e) = transport.connect() {
                    trace!("Connection attempt failed: {e}");
                }
            }
        }
    }

    pub fn registry(&self) -> &SendableComponentRegistry {
        &self.component_registry
    }

    pub fn registry_mut(&mut self) -> &mut SendableComponentRegistry {
        &mut self.component_registry
    }

    /// Check if sending is enabled
    pub fn is_enabled(&self) -> bool {
        self.transport.is_some()
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        if let Some(ref t) = self.transport {
            return t.is_connected();
        }
        false
    }

    /// Get the current frame number
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Begin a new frame. All messages sent until `end_frame()` will be
    /// grouped together and the receiver will apply them atomically.
    pub fn start_frame(&mut self) {
        let frame_num = self.frame_number;
        self.frame_number += 1;
        self.in_frame = true;

        // Send frame begin marker
        self.send_message_immediate(SceneMessage::FrameBegin { frame_number: frame_num });
    }

    /// End the current frame and flush all accumulated messages.
    pub fn end_frame(&mut self) {
        if !self.in_frame {
            return;
        }

        let frame_num = self.frame_number.saturating_sub(1);
        self.in_frame = false;

        // Flush any batched messages
        let messages: Vec<SceneMessage> = std::mem::take(&mut self.batch_buffer);

        // Send batched messages if any
        if !messages.is_empty() {
            let batch = if messages.len() == 1 {
                messages.into_iter().next().unwrap()
            } else {
                SceneMessage::Batch(messages)
            };
            self.send_message_immediate(batch);
        }

        // Send frame end marker
        self.send_message_immediate(SceneMessage::FrameEnd { frame_number: frame_num });
    }

    /// Send a message immediately (bypassing frame batching)
    ///
    /// # Errors
    ///
    /// Returns `TransportError` if can't send the message.
    fn send_message_immediate(&mut self, message: SceneMessage) {
        if !self.is_enabled() {
            return;
        }

        // Periodically attempt to reconnect if not connected
        self.maybe_reconnect();

        let sequence = self.sequence;
        self.sequence += 1;
        let frame = MessageFrame::new(sequence, message);

        if let Some(ref t) = self.transport {
            match t.send(&frame) {
                Ok(()) => {
                    trace!("Sent message with sequence {sequence}");
                }
                Err(TransportError::NotConnected | TransportError::ConnectionClosed) => {
                    // Silently drop message - receiver not available
                    trace!("Message dropped - no receiver connected");
                }
                Err(e) => {
                    error!("Failed to send message: {e}"); // Don't fail the render loop, just log error and continue
                }
            }
        } else {
            // No transport set, silently drop
            trace!("Message dropped - no transport configured");
        }
    }

    /// Send a raw message (respects frame batching)
    pub fn send_message(&mut self, message: SceneMessage) {
        if !self.is_enabled() {
            return;
        }

        // If in a frame, batch the message
        if self.in_frame {
            self.batch_buffer.push(message);
            return;
        }

        self.send_message_immediate(message);
    }

    /// Send an entity spawn notification
    pub fn send_entity_spawn(&mut self, name: &str, is_renderable: bool) {
        let message = SceneMessage::EntitySpawn {
            entity_name: name.to_string(),
            is_renderable,
        };
        self.send_message(message);
    }

    /// Get a reference to the component registry
    pub fn component_registry(&self) -> &SendableComponentRegistry {
        &self.component_registry
    }

    /// Get a mutable reference to the component registry
    pub fn component_registry_mut(&mut self) -> &mut SendableComponentRegistry {
        &mut self.component_registry
    }

    /// Check if a component type is registered as network sendable
    pub fn is_component_sendable<T: 'static>(&self) -> bool {
        self.component_registry.is_network_sendable::<T>()
    }

    /// Try to serialize a component using the registry
    pub fn try_serialize_component(&self, type_id: &ComponentTypeId, component: &dyn std::any::Any) -> Option<Box<dyn NetworkSendable>> {
        self.component_registry.try_serialize_component(type_id, component)
    }

    /// Send a component insertion
    ///
    /// # Errors
    ///
    /// Returns `TransportError` if can't serialize the component.
    #[allow(clippy::needless_pass_by_value)]
    pub fn send_component(&mut self, entity_name: &str, component: Box<dyn NetworkSendable>) -> Result<(), TransportError> {
        let data = component.serialize_to_bytes().map_err(TransportError::Serialization)?;
        let message = SceneMessage::ComponentInsert {
            entity_name: entity_name.to_string(),
            component_type: component.component_type_id(),
            data,
        };
        self.send_message(message);
        Ok(())
    }

    /// Close the connection
    pub fn close(&mut self) {
        if let Some(ref t) = self.transport {
            t.close();
        }
    }
}

impl Default for SceneSender {
    fn default() -> Self {
        Self::new(TransportConfig::default())
    }
}
impl Drop for SceneSender {
    fn drop(&mut self) {
        self.close();
    }
}
