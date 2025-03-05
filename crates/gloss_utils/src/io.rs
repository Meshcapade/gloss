//from https://github.com/takahirox/wgpu-rust-renderer/blob/main/src/utils/file_loader.rs#L10
pub struct FileLoader {}

// Non-Wasm
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;

#[cfg(not(target_arch = "wasm32"))]
impl FileLoader {
    /// # Panics
    /// Will panic if the path cannot be opened
    #[allow(clippy::unused_async)] //we want to maintain the same code signature as the wasm version which needs
                                   // async
    pub async fn open(file_path: &str) -> File {
        File::open(file_path).unwrap()
    }
}

// Wasm
#[cfg(target_arch = "wasm32")]
use {
    std::io::Cursor,
    wasm_bindgen::JsCast,
    wasm_bindgen_futures::JsFuture,
    web_sys::{Request, RequestInit, RequestMode, Response},
};

// @TODO: Proper error handling
#[cfg(target_arch = "wasm32")]
impl FileLoader {
    pub async fn open(file_path: &str) -> Cursor<Vec<u8>> {
        let result = fetch_as_binary(file_path).await.unwrap();
        Cursor::new(result)
    }
}

// @TODO: Proper error handling
#[cfg(target_arch = "wasm32")]
pub async fn fetch_as_binary(url: &str) -> Result<Vec<u8>, String> {
    let mut opts = RequestInit::new();
    #[allow(deprecated)]
    opts.method("GET");
    #[allow(deprecated)]
    opts.mode(RequestMode::Cors); // @TODO: Should be able to opt-out

    let request = match Request::new_with_str_and_init(&url, &opts) {
        Ok(request) => request,
        Err(_e) => return Err("Failed to create request".to_string()),
    };

    let window = web_sys::window().unwrap();
    let response = match JsFuture::from(window.fetch_with_request(&request)).await {
        Ok(response) => response,
        Err(_e) => return Err("Failed to fetch".to_string()),
    };

    let response: Response = match response.dyn_into() {
        Ok(response) => response,
        Err(_e) => return Err("Failed to dyn_into Response".to_string()),
    };

    let buffer = match response.array_buffer() {
        Ok(buffer) => buffer,
        Err(_e) => return Err("Failed to get as array buffer".to_string()),
    };

    let buffer = match JsFuture::from(buffer).await {
        Ok(buffer) => buffer,
        Err(_e) => return Err("Failed to ...?".to_string()),
    };

    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

// https://www.reddit.com/r/rust/comments/11hcyv4/best_way_of_associating_enums_with_values/jav1kgf/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
/// associating a extension with a enum
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
#[derive(Debug, EnumIter)]
pub enum FileType {
    Obj,
    Ply,
    Unknown,
}
impl FileType {
    pub fn value(&self) -> &'static [&'static str] {
        match self {
            Self::Obj => &["obj"],
            Self::Ply => &["ply"],
            // Self::Ply => &["ply"],
            Self::Unknown => &[""],
        }
    }
    pub fn find_match(ext: &str) -> Self {
        Self::iter()
            .find(|filetype| filetype.value().contains(&(ext.to_lowercase()).as_str()))
            .unwrap_or(FileType::Unknown)
    }
}
