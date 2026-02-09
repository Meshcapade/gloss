/// `SceneReceiver`: Networked scene/component receiver for remote visualization.
///
/// The receiver listens for incoming messages and applies them to the local scene.
/// It's usually paired with the `ReceiverPlugin` which makes sure to tool and receive and apply the messages correctly.
///
/// # Usage Example
///
/// ```ignore
/// use gloss_renderer::network::{SceneReceiver, TransportConfig};
///
/// // Configure receiver (address/port should match sender)
/// let mut receiver = SceneReceiver::new(TransportConfig {
///     address: "127.0.0.1".to_string(),
///     port: 46377,
///     ..Default::default()
/// });
///
/// // Optionally register custom serializable components
/// receiver.registry_mut().register_component_simple::<SerializableVerts, Verts>();
///
/// // Connect to sender (client mode)
/// receiver.connect_to_sender()?;
///
/// // Add as resource to your ECS world or viewer
/// viewer.add_resource(receiver);
///
/// // Optionally, insert the SceneReceiverPlugin to poll for messages each frame
/// viewer.insert_plugin(SceneReceiverPlugin::new(true));
/// ```
///
/// # Notes
/// - Use batching on the sender side to ensure atomic scene updates.
/// - Only components registered and implementing `NetworkReceivable` can be deserialized and applied.
/// - The receiver can be polled manually or via a plugin for new messages.
/// - See the Python `scene_receiver.py` for an example workflow
use super::messages::{MessageFrame, SceneMessage};
use super::transport::{Transport, TransportConfig, TransportError};
use crate::components::{Name, OnGui, Renderable};
use crate::network::component_registry::ReceivableComponentRegistry;
use crate::scene::Scene;
use gloss_hecs::{CommandBuffer, EntityBuilder};
use log::{info, warn};
use std::sync::Arc;

#[derive(Clone)]
pub struct SceneReceiver {
    /// The underlying transport for receiving messages
    transport: Option<Arc<dyn Transport>>,
    /// Last received sequence number (for ordering/gap detection)
    last_sequence: u64,
    /// Transport configuration
    config: TransportConfig,
    /// Component registry for deserialization
    registry: ReceivableComponentRegistry,
    /// Last connection attempt time
    last_connect_attempt: Option<std::time::Instant>,
}

impl SceneReceiver {
    /// Create a new scene receiver (not connected)
    pub fn new(config: TransportConfig) -> Self {
        let mut registry = ReceivableComponentRegistry::new();
        registry.register_default_components();

        Self {
            transport: None,
            last_sequence: 0,
            config,
            registry,
            last_connect_attempt: None,
        }
    }

    pub fn registry(&self) -> &ReceivableComponentRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut ReceivableComponentRegistry {
        &mut self.registry
    }

    /// Set the transport
    pub fn set_transport(&mut self, transport: Arc<dyn Transport>) {
        self.transport = Some(transport);
    }

    /// Connect to a remote sender as a client (instead of listening as a server).
    /// This makes the receiver connect TO the sender, reversing the typical roles.
    /// Use this when you want the sender to be the server and receiver to be the client.
    ///
    /// # Errors
    /// Returns `TransportError` if creating the TCP receiver or establishing the connection fails.
    pub fn connect_to_sender(&mut self) -> Result<(), TransportError> {
        use super::transport::TcpClient;

        // Check if we should attempt reconnection (rate limiting)
        if !self.should_attempt_reconnect() {
            return Err(TransportError::NotConnected);
        }

        self.last_connect_attempt = Some(std::time::Instant::now());

        let addr = format!("{}:{}", self.config.address, self.config.port);
        let tcp_client = TcpClient::new(&addr)?;
        tcp_client.connect()?;
        self.set_transport(Arc::new(tcp_client));
        log::info!("Successfully connected to sender at {addr}");
        Ok(())
    }

    /// Check if we should attempt to reconnect (rate limiting)
    fn should_attempt_reconnect(&self) -> bool {
        if !self.config.auto_reconnect {
            return false;
        }

        if self.is_connected() {
            return false;
        }

        match self.last_connect_attempt {
            None => true, // Never attempted
            Some(last_attempt) => {
                let reconnect_delay = std::time::Duration::from_millis(self.config.reconnect_delay_ms);
                last_attempt.elapsed() >= reconnect_delay
            }
        }
    }

    /// Check if receiving is enabled
    pub fn is_enabled(&self) -> bool {
        self.transport.is_some()
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        if let Some(ref transport) = self.transport {
            transport.is_connected()
        } else {
            false
        }
    }

    /// Receive all pending frames and return them for processing.
    /// This separates frame reception from scene processing to avoid borrowing conflicts.
    ///
    /// # Errors
    ///
    /// Returns `TransportError` if the underlying transport returns an error while
    /// attempting to receive frames.
    pub fn receive_pending_frames(&mut self) -> Result<Vec<crate::network::MessageFrame>, TransportError> {
        if !self.is_enabled() {
            println!("Receiver not enabled, transport is not set, skipping receive.");
            return Ok(Vec::new());
        }

        let mut frames = Vec::new();

        if let Some(ref transport) = self.transport {
            loop {
                match transport.receive(false) {
                    Ok(Some(frame)) => {
                        frames.push(frame);
                        //if we buffered 30 frames we just return then so we apply them ,otherwise this loop will get stuck here receiving frames
                        if frames.len() >= 30 {
                            warn!("Filled the maximum buffer of 30 frames, returning to apply them");
                            break;
                        }
                    }
                    Ok(None) => {
                        // No more frames available
                        break;
                    }
                    Err(TransportError::ConnectionClosed) => {
                        // Connection was lost, we need to clear transport to allow reconnection
                        log::warn!("Connection lost, clearing transport for reconnection");
                        self.transport = None;
                        return Err(TransportError::ConnectionClosed);
                    }
                    Err(e) => {
                        // Other errors are passed through
                        return Err(e);
                    }
                }
            }
        }

        Ok(frames)
    }

    /// Process a received frame
    pub fn process_frame(&mut self, scene: &Scene, command_buffer: &mut CommandBuffer, frame: &MessageFrame) {
        // Track sequence for gap detection
        if frame.sequence > 0 && frame.sequence != self.last_sequence + 1 {
            warn!("Sequence gap detected: expected {}, got {}", self.last_sequence + 1, frame.sequence);
        }
        self.last_sequence = frame.sequence;

        // Process the message
        self.process_message(scene, command_buffer, &frame.message);
    }

    /// Process a single message
    fn process_message(&mut self, scene: &Scene, command_buffer: &mut CommandBuffer, message: &SceneMessage) {
        match message {
            SceneMessage::Batch(messages) => {
                for msg in messages {
                    self.process_message(scene, command_buffer, msg);
                }
            }

            SceneMessage::EntitySpawn {
                entity_name,
                is_renderable,
                is_on_gui,
            } => {
                // info!("Received EntitySpawn: {} ({})", entity_name, is_renderable);

                //skip because we want the floor to be created by the local viewer
                if entity_name == "floor" || entity_name == "gloss_camera" {
                    return;
                }

                if scene.get_entity_with_name(entity_name).is_some() {
                    warn!("Entity with name '{entity_name}' already exists, skipping spawn");
                    return;
                }

                let mut entity_builder = EntityBuilder::new();
                let mut entity_builder = entity_builder.add(Name(entity_name.to_string()));
                if *is_renderable {
                    entity_builder = entity_builder.add(Renderable);
                }
                if *is_on_gui {
                    entity_builder = entity_builder.add(OnGui);
                }

                command_buffer.spawn(entity_builder.build());
            }

            SceneMessage::ComponentInsert {
                entity_name,
                component_type,
                data,
            } => {
                let ent = if entity_name == "entity_resource" {
                    Some(scene.get_entity_resource())
                } else {
                    scene.get_entity_with_name(entity_name)
                };

                if let Some(entity) = ent {
                    // Apply the component using the registry
                    if let Err(e) = self.registry.apply_component(command_buffer, entity, component_type, data) {
                        warn!("Failed to apply component {}: {}", component_type.0, e);
                    }
                } else {
                    warn!("Entity with name '{entity_name}' not found for ComponentInsert");
                }
            }

            SceneMessage::FrameBegin { frame_number } => {
                info!("Received FrameBegin: {frame_number}");
                // Note: Frame buffering is handled at the higher level (receive_complete_frame)
                // If we get here, we're processing messages immediately without frame boundaries
            }

            SceneMessage::FrameEnd { frame_number } => {
                info!("Received FrameEnd: {frame_number}");
                // self.last_applied_frame = frame_number;
            }

            SceneMessage::Ping { timestamp_ms } => {
                info!("Received Ping: {timestamp_ms}");
                // Optionally respond with Pong (would need sender capability)
            }

            SceneMessage::Pong { timestamp_ms } => {
                info!("Received Pong: {timestamp_ms}");
                // Calculate latency if needed
            }
        }
    }

    /// Close the connection
    pub fn close(&mut self) {
        if let Some(ref transport) = self.transport {
            transport.close();
        }
    }
}

impl Default for SceneReceiver {
    fn default() -> Self {
        Self::new(TransportConfig::default())
    }
}
