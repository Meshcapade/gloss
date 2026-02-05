//! Receiver plugin for network synchronization.
//!
//! This module provides a logic system that can be added to the renderer's
//! plugin manager. It will poll for network messages at the beginning of
//! each frame and apply them to the scene.

use super::receiver::SceneReceiver;
use crate::network::SceneMessage;
use crate::plugin_manager::systems::LogicSystem;
use crate::plugin_manager::{runner::RunnerState, Plugin};
use crate::plugin_manager::{EventSystem, GuiSystem, InitSystem};
use crate::scene::Scene;
use gloss_hecs::CommandBuffer;
use log::error;

#[allow(clippy::empty_line_after_doc_comments)]
/// The logic system function that processes network messages.
extern "C" fn network_receiver_system(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    let frames = if let Ok(mut receiver) = scene.get_resource::<&mut SceneReceiver>() {
        // Try to connect if not connected (with rate limiting)
        if !receiver.is_connected() {
            match receiver.connect_to_sender() {
                Ok(()) => {
                    log::info!("Successfully reconnected to sender");
                }
                Err(super::transport::TransportError::NotConnected) => {
                    // Rate limited or auto_reconnect disabled, silently skip
                }
                Err(e) => {
                    // Connection failed for other reasons
                    log::debug!("Failed to connect to sender: {e}");
                }
            }
        }

        // Receive all pending frames if connected
        if receiver.is_connected() {
            match receiver.receive_pending_frames() {
                Ok(frames) => frames,
                Err(super::transport::TransportError::ConnectionClosed) => {
                    // Connection was lost, receiver was automatically cleared
                    // Next frame will attempt to reconnect
                    log::info!("Connection lost, will attempt to reconnect");
                    Vec::new()
                }
                Err(e) => {
                    error!("Error receiving network frames: {e}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    //process each message individually so that after each message we apply the command buffer to the scene. This is in order to prevent having a batch of messages that both spawn a entity and also try to insert components into it
    for frame in frames {
        let individual_frames = frame.to_individual_messages();
        for individual_frame in individual_frames {
            if let Ok(mut receiver) = scene.get_resource::<&mut SceneReceiver>() {
                //if the message is a batch,we apply each message one by one
                receiver.process_frame(scene, &mut command_buffer, &individual_frame);
            }
            // println!("Applying command buffer with {} commands", command_buffer.len());
            scene.world_mut().run_command_buffer(&mut command_buffer);
            //if we spawned a entity, we have to update the name2entity in the scene because the name2entity was not updated since we spawned using a command buffer
            if let SceneMessage::EntitySpawn { entity_name, .. } = &individual_frame.message {
                scene.update_name2entity(entity_name);
            }
        }
    }
}

#[derive(Clone)]
pub struct SceneReceiverPlugin {
    pub autorun: bool,
}
impl SceneReceiverPlugin {
    pub fn new(autorun: bool) -> Self {
        Self { autorun }
    }
}

impl Plugin for SceneReceiverPlugin {
    fn autorun(&self) -> bool {
        self.autorun
    }
    fn init_system(&self) -> Option<InitSystem> {
        None
    }
    #[allow(clippy::vec_init_then_push)]
    fn event_systems(&self) -> Vec<EventSystem> {
        Vec::new()
    }
    fn logic_systems(&self) -> Vec<LogicSystem> {
        vec![
            LogicSystem::new(network_receiver_system).with_name("network_receive"), // Automatically receive components from the network
        ]
    }

    #[allow(clippy::vec_init_then_push)]
    fn gui_systems(&self) -> Vec<GuiSystem> {
        Vec::new()
    }
}
