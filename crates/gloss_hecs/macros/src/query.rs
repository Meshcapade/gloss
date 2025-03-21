use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{DeriveInput, Error, Ident, Lifetime, Result, Type};

pub fn derive(input: DeriveInput) -> Result<TokenStream2> {
    let ident = input.ident;
    let vis = input.vis;
    let data = match input.data {
        syn::Data::Struct(s) => s,
        _ => return Err(Error::new_spanned(ident, "derive(Query) may only be applied to structs")),
    };
    let lifetime = input.generics.lifetimes().next().map(|x| x.lifetime.clone());
    let lifetime = match lifetime {
        Some(x) => x,
        None => return Err(Error::new_spanned(input.generics, "must have exactly one lifetime parameter")),
    };
    if input.generics.params.len() != 1 {
        return Err(Error::new_spanned(
            ident,
            "must have exactly one lifetime parameter and no type parameters",
        ));
    }

    let (fields, fetches) = match data.fields {
        syn::Fields::Named(ref fields) => fields
            .named
            .iter()
            .map(|f| (syn::Member::Named(f.ident.clone().unwrap()), query_fetch_ty(&lifetime, &f.ty)))
            .unzip(),
        syn::Fields::Unnamed(ref fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| {
                (
                    syn::Member::Unnamed(syn::Index {
                        index: i as u32,
                        span: Span::call_site(),
                    }),
                    query_fetch_ty(&lifetime, &f.ty),
                )
            })
            .unzip(),
        syn::Fields::Unit => (Vec::new(), Vec::new()),
    };
    let fetches = fetches.into_iter().collect::<Vec<_>>();
    let fetch_ident = Ident::new(&format!("__HecsInternal{}Fetch", ident), Span::call_site());
    let fetch = match data.fields {
        syn::Fields::Named(_) => quote! {
            #vis struct #fetch_ident {
                #(
                    #fields: #fetches,
                )*
            }
        },
        syn::Fields::Unnamed(_) => quote! {
            #vis struct #fetch_ident(#(#fetches),*);
        },
        syn::Fields::Unit => quote! {
            #vis struct #fetch_ident;
        },
    };
    let state_ident = Ident::new(&format!("__HecsInternal{}State", ident), Span::call_site());
    let state = match data.fields {
        syn::Fields::Named(_) => quote! {
            #[derive(Clone, Copy)]
            #vis struct #state_ident<'a> {
                #(
                    #fields: <#fetches as ::gloss_hecs::Fetch<'a>>::State,
                )*
            }
        },
        syn::Fields::Unnamed(_) => quote! {
            #[derive(Clone, Copy)]
            #vis struct #state_ident<'a>(#(<#fetches as ::gloss_hecs::Fetch<'a>>::State),*);
        },
        syn::Fields::Unit => quote! {
            #[derive(Clone, Copy)]
            #vis struct #state_ident;
        },
    };

    Ok(quote! {
        impl<'a> ::gloss_hecs::Query for #ident<'a> {
            type Fetch = #fetch_ident;
        }

        #[doc(hidden)]
        #fetch

        #[doc(hidden)]
        #state

        unsafe impl<'a> ::gloss_hecs::Fetch<'a> for #fetch_ident {
            type Item = #ident<'a>;

            type State = #state_ident<'a>;

            fn dangling() -> Self {
                Self {
                    #(
                        #fields: #fetches::dangling(),
                    )*
                }
            }

            #[allow(unused_variables, unused_mut)]
            fn access(archetype: &::gloss_hecs::Archetype) -> ::std::option::Option<::gloss_hecs::Access> {
                let mut access = ::gloss_hecs::Access::Iterate;
                #(
                    access = ::core::cmp::max(access, #fetches::access(archetype)?);
                )*
                ::std::option::Option::Some(access)
            }

            #[allow(unused_variables)]
            fn borrow(archetype: &::gloss_hecs::Archetype, state: Self::State) {
                #(#fetches::borrow(archetype, state.#fields);)*
            }

            #[allow(unused_variables)]
            fn prepare(archetype: &::gloss_hecs::Archetype) -> ::std::option::Option<Self::State> {
                ::std::option::Option::Some(#state_ident {
                    #(
                        #fields: #fetches::prepare(archetype)?,
                    )*
                })
            }

            #[allow(unused_variables)]
            fn execute(archetype: &'a ::gloss_hecs::Archetype, state: Self::State) -> Self {
                Self {
                    #(
                        #fields: #fetches::execute(archetype, state.#fields),
                    )*
                }
            }

            #[allow(unused_variables)]
            fn release(archetype: &::gloss_hecs::Archetype, state: Self::State) {
                #(#fetches::release(archetype, state.#fields);)*
            }

            #[allow(unused_variables, unused_mut)]
            fn for_each_borrow(mut f: impl ::core::ops::FnMut(::gloss_hecs::StableTypeId, bool)) {
                #(
                    <#fetches as ::gloss_hecs::Fetch<'static>>::for_each_borrow(&mut f);
                )*
            }

            #[allow(unused_variables)]
            unsafe fn get(&self, n: usize) -> Self::Item {
                #ident {
                    #(
                        #fields: <#fetches as ::gloss_hecs::Fetch<'a>>::get(&self.#fields, n),
                    )*
                }
            }
        }
    })
}

fn query_fetch_ty(lifetime: &Lifetime, ty: &Type) -> TokenStream2 {
    struct Visitor<'a> {
        replace: &'a Lifetime,
    }
    impl syn::visit_mut::VisitMut for Visitor<'_> {
        fn visit_lifetime_mut(&mut self, l: &mut Lifetime) {
            if l == self.replace {
                *l = Lifetime::new("'static", Span::call_site());
            }
        }
    }

    let mut ty = ty.clone();
    syn::visit_mut::visit_type_mut(&mut Visitor { replace: lifetime }, &mut ty);
    quote! {
        <#ty as ::gloss_hecs::Query>::Fetch
    }
}
