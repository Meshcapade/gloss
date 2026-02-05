use super::messages::ComponentTypeId;
use crate::components::{Colors, Edges, Faces, Normals, UVs, Verts, VisLines, VisMesh, VisPoints};
use crate::network::serializable_components::{
    SerializableColors, SerializableEdges, SerializableFaces, SerializableNormals, SerializableUVs, SerializableVerts, SerializableVisLines,
    SerializableVisMesh, SerializableVisPoints,
};
use crate::network::{FromSerializable, NetworkSendable};
use gloss_hecs::{CommandBuffer, Entity};
use std::collections::HashMap;

// =====================================================================
// ==========================  Sendable  ===============================
// =====================================================================

/// Type alias for a serialization function that takes a component reference and returns a serializable object
type SendableSerializationFn = fn(&dyn std::any::Any) -> Option<Box<dyn NetworkSendable>>;

/// Registry for component serializers for sending over the network.
///
/// This registry maps component `TypeIds` to their serialization functions,
/// allowing us to determine at runtime which components should be sent
/// over the network and how to serialize them.
#[derive(Clone)]
pub struct SendableComponentRegistry {
    /// Map from component `TypeId` to serialization functions
    component_serializers: HashMap<ComponentTypeId, SendableSerializationFn>,
}

impl SendableComponentRegistry {
    pub fn new() -> Self {
        Self {
            component_serializers: HashMap::new(),
        }
    }

    /// Register a component type and the corresponding network sendable type.
    pub fn register_component_simple<C, S>(&mut self)
    where
        C: crate::network::serializable_components::ToSerializable<S> + 'static,
        S: NetworkSendable + 'static,
    {
        let serialize_fn = |any: &dyn std::any::Any| any.downcast_ref::<C>().map(|v| Box::new(v.to_serializable()) as Box<dyn NetworkSendable>);
        let base_type_id = ComponentTypeId::of::<C>();
        self.component_serializers.insert(base_type_id, serialize_fn);
    }

    /// Check if a component type is registered as network sendable
    pub fn is_network_sendable<T: 'static>(&self) -> bool {
        self.component_serializers.contains_key(&ComponentTypeId::of::<T>())
    }

    /// Try to serialize a component of the given type ID
    pub fn try_serialize_component(&self, type_id: &ComponentTypeId, component: &dyn std::any::Any) -> Option<Box<dyn NetworkSendable>> {
        if let Some(serializable_func) = self.component_serializers.get(type_id) {
            return (serializable_func)(component);
        }
        None
    }

    /// Initialize with default network sendable component registrations
    pub fn with_default_components() -> Self {
        let mut registry = Self::new();
        registry.register_default_components();
        registry
    }

    /// Register all the default network sendable components
    pub fn register_default_components(&mut self) {
        // Register CPU components that can be sent over network
        self.register_component_simple::<Verts, SerializableVerts>();
        self.register_component_simple::<Faces, SerializableFaces>();
        self.register_component_simple::<UVs, SerializableUVs>();
        self.register_component_simple::<Normals, SerializableNormals>();
        self.register_component_simple::<Colors, SerializableColors>();
        self.register_component_simple::<Edges, SerializableEdges>();
        self.register_component_simple::<VisLines, SerializableVisLines>();
        self.register_component_simple::<VisPoints, SerializableVisPoints>();
        self.register_component_simple::<VisMesh, SerializableVisMesh>();
    }
}

impl Default for SendableComponentRegistry {
    fn default() -> Self {
        Self::with_default_components()
    }
}

// =====================================================================
// =========================  Receivable  ==============================
// =====================================================================

/// Type alias for component applicator functions.
/// These functions take serialized data and apply it to an entity in the scene.
type ReceivableComponentApplicator = fn(&mut CommandBuffer, Entity, &[u8]) -> Result<(), String>;

/// Registry for component deserializers and applicators.
///
/// Since we can't know at compile time which components will be received,
/// we need a runtime registry that maps component type IDs to their
/// deserialization and application logic.
/// The component ID corresponds to the serialized one and the function pointer will take serialized data, deserialize it, and insert it to an entity.
#[derive(Clone)]
pub struct ReceivableComponentRegistry {
    /// Map from component type ID to applicator function
    component_applicators: HashMap<ComponentTypeId, ReceivableComponentApplicator>,
}

impl ReceivableComponentRegistry {
    pub fn new() -> Self {
        Self {
            component_applicators: HashMap::new(),
        }
    }

    /// Register a component type that can be directly deserialized with bincode.
    /// This is for serializable types that need to be converted to their actual component types.
    /// S corresponds to the serialized struct and C to the actual component
    pub fn register_component_simple<S, C>(&mut self)
    where
        S: crate::network::NetworkReceivable + 'static,
        C: gloss_hecs::Component + FromSerializable<S> + 'static,
    {
        // Create a function pointer that can deserialize and apply the specific component type
        fn apply_component<S, C>(command_buffer: &mut CommandBuffer, entity: Entity, data: &[u8]) -> Result<(), String>
        where
            S: crate::network::NetworkReceivable + 'static,
            C: gloss_hecs::Component + FromSerializable<S> + 'static,
        {
            let config = bincode::config::standard();
            let (serializable_component, _): (S, usize) =
                bincode::serde::decode_from_slice(data, config).map_err(|e| format!("Failed to decode component: {e}"))?;

            let component = C::from_serializable(&serializable_component);
            command_buffer.insert_one(entity, component);
            Ok(())
        }

        let type_id = ComponentTypeId::of::<S>();
        self.component_applicators.insert(type_id, apply_component::<S, C>);
    }

    /// Apply a component to an entity using the registered applicator.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if no applicator is registered for the provided component
    /// type, or if the registered applicator itself fails; the `Err` contains a
    /// human-readable description of the failure.
    pub fn apply_component(&self, command_buffer: &mut CommandBuffer, entity: Entity, type_id: &ComponentTypeId, data: &[u8]) -> Result<(), String> {
        if let Some(entry) = self.component_applicators.get(type_id) {
            (entry)(command_buffer, entity, data)
        } else {
            Err(format!("No applicator registered for component type: {}", type_id.0))
        }
    }

    /// Initialize with default network receivable component registrations
    pub fn with_default_components() -> Self {
        let mut registry = Self::new();
        registry.register_default_components();
        registry
    }

    /// Register all the default network receivable components
    pub fn register_default_components(&mut self) {
        // Register CPU components that can be received over network
        self.register_component_simple::<SerializableVerts, Verts>();
        self.register_component_simple::<SerializableFaces, Faces>();
        self.register_component_simple::<SerializableUVs, UVs>();
        self.register_component_simple::<SerializableNormals, Normals>();
        self.register_component_simple::<SerializableColors, Colors>();
        self.register_component_simple::<SerializableEdges, Edges>();
        self.register_component_simple::<SerializableVisLines, VisLines>();
        self.register_component_simple::<SerializableVisPoints, VisPoints>();
        self.register_component_simple::<SerializableVisMesh, VisMesh>();
    }
}

impl Default for ReceivableComponentRegistry {
    fn default() -> Self {
        Self::with_default_components()
    }
}
