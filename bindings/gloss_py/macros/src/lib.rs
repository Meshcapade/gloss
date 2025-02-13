extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{self, Ident, Type};

#[proc_macro_derive(PyComponent)]
pub fn pycomponent_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let body: &syn::Data = &ast.data;

    // inspiration from venial
    // https://stackoverflow.com/questions/73678625/how-do-i-extract-information-about-the-type-in-a-derive-macro

    // Ensure it's deriving for a struct.
    let s = match body {
        syn::Data::Struct(s) => s,
        _ => panic!("Can only derive this trait on a struct"),
    };

    // Get the struct's first field.
    let fields = &s.fields;
    let named_fields = match fields {
        syn::Fields::Named(named_fields) => named_fields,
        _ => panic!("Expected a named field"),
    };

    let inners = &named_fields.named;
    if inners.len() != 1 {
        panic!("Expected exactly one named field");
    }

    // Get the name and type of the first field.
    // let first_field_name = &inners.first().unwrap().ident;
    let first_field_ty = &inners.first().unwrap().ty;

    let gen = quote! {
        // impl PyComponent for #name {
        #[pymethods]
        impl #name {
            //ideally we would use a function that takes PyEntityMut as parameter. However, PyEntityMut cannot be shared between multiple crates due to this:
            // https://github.com/PyO3/pyo3/issues/1444
            // therefore we pass around just the raw data for the entity and the scene_ptr
            pub fn insert_to_entity(&mut self, entity_bits: u64, scene_ptr_idx: u64) {
                let entity = Entity::from_bits(entity_bits).unwrap();
                let scene_ptr = scene_ptr_idx as *mut Scene;
                let scene: &mut Scene = unsafe { &mut *scene_ptr };
                scene.world
                    .insert_one(entity, self.inner.clone())
                    .ok();
            }
            #[staticmethod]
            pub fn get(entity_bits: u64, scene_ptr_idx: u64) -> Self {
                let entity = Entity::from_bits(entity_bits).unwrap();
                //TODO this is super brittle because if the scene obj is ever compiled differently in gloss, any plugin that depends on derefering it will fail because the Scene object will have different size
                let scene_ptr = scene_ptr_idx as *mut Scene;
                let scene: &mut Scene = unsafe { &mut *scene_ptr };
                let comp = scene.get_comp::<&mut #first_field_ty>(&entity).unwrap();
                Self {
                    inner: comp.clone(),
                }
            }
            #[staticmethod]
            pub fn exists(entity_bits: u64, scene_ptr_idx: u64) -> bool {
                let entity = Entity::from_bits(entity_bits).unwrap();
                //TODO this is super brittle because if the scene obj is ever compiled differently in gloss, any plugin that depends on derefering it will fail because the Scene object will have different size
                let scene_ptr = scene_ptr_idx as *mut Scene;
                let scene: &mut Scene = unsafe { &mut *scene_ptr };
                scene.world.has::<#first_field_ty>(entity).unwrap()
            }
            #[staticmethod]
            pub fn remove(entity_bits: u64, scene_ptr_idx: u64) {
                let entity = Entity::from_bits(entity_bits).unwrap();
                //TODO this is super brittle because if the scene obj is ever compiled differently in gloss, any plugin that depends on derefering it will fail because the Scene object will have different size
                let scene_ptr = scene_ptr_idx as *mut Scene;
                let scene: &mut Scene = unsafe { &mut *scene_ptr };
                scene.world.remove_one::<#first_field_ty>(entity).ok(); //don't unwrap because we don't care if the component exists or not
            }
        }
    };
    gen.into()
}

#[proc_macro_derive(PtrDeref)]
pub fn ptr_deref_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let body: &syn::Data = &ast.data;

    // inspiration from venial
    // https://stackoverflow.com/questions/73678625/how-do-i-extract-information-about-the-type-in-a-derive-macro
    // Ensure it's deriving for a struct.
    let s = match body {
        syn::Data::Struct(s) => s,
        _ => panic!("Can only derive this trait on a struct"),
    };

    // Get the struct's first field.
    let fields = &s.fields;
    let named_fields = match fields {
        syn::Fields::Named(named_fields) => named_fields,
        _ => panic!("Expected a named field"),
    };

    let inners = &named_fields.named;
    if inners.len() != 1 {
        panic!("Expected exactly one named field");
    }

    // Get the name and type of the first field.
    let first_field_name = inners.first().unwrap().ident.as_ref().unwrap();
    let first_field_ty = &inners.first().unwrap().ty;
    let typeptr = match first_field_ty {
        syn::Type::Ptr(typeptr) => typeptr,
        _ => panic!("Expected a type to be a ptr, it is {:?}", first_field_ty),
    };
    let elem_type = &*typeptr.elem;

    impl_ptr_deref_macro(name, elem_type, first_field_name)
}

#[proc_macro_derive(DirectDeref)]
pub fn direct_deref_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let body: &syn::Data = &ast.data;

    // inspiration from venial
    // https://stackoverflow.com/questions/73678625/how-do-i-extract-information-about-the-type-in-a-derive-macro
    // Ensure it's deriving for a struct.
    let s = match body {
        syn::Data::Struct(s) => s,
        _ => panic!("Can only derive this trait on a struct"),
    };

    // Get the struct's first field.
    let fields = &s.fields;
    let named_fields = match fields {
        syn::Fields::Named(named_fields) => named_fields,
        _ => panic!("Expected a named field"),
    };

    let inners = &named_fields.named;
    if inners.len() != 1 {
        panic!("Expected exactly one named field");
    }

    // Get the name and type of the first field.
    let first_field_name = inners.first().unwrap().ident.as_ref().unwrap();
    let first_field_ty = &inners.first().unwrap().ty;
    let typeinner = match first_field_ty {
        syn::Type::Path(typeinner) => typeinner,
        _ => panic!("Expected a type to be a ptr, it is {:?}", first_field_ty),
    };
    let elem_type = &typeinner.path.segments.first().unwrap().ident;

    impl_direct_deref_macro(name, elem_type, first_field_name)
}

fn impl_ptr_deref_macro(struct_name: &Ident, elem_type: &Type, first_field_name: &Ident) -> TokenStream {
    let gen = quote! {
        impl std::ops::Deref for #struct_name {
            type Target = #elem_type;
            fn deref(&self) -> &Self::Target {
                unsafe { &*self.#first_field_name }
            }
        }
        impl std::ops::DerefMut for #struct_name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *self.#first_field_name }
            }
        }
    };
    gen.into()
}

fn impl_direct_deref_macro(struct_name: &Ident, elem_type: &Ident, first_field_name: &Ident) -> TokenStream {
    let gen = quote! {
        impl std::ops::Deref for #struct_name {
            type Target = #elem_type;
            fn deref(&self) -> &Self::Target {
                unsafe { &self.#first_field_name }
            }
        }
        impl std::ops::DerefMut for #struct_name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut self.#first_field_name }
            }
        }
    };
    gen.into()
}
