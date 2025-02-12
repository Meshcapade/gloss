use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(msg: String);

    type Error;

    #[wasm_bindgen(constructor)]
    fn new() -> Error;

    #[wasm_bindgen(structural, method, getter)]
    fn stack(error: &Error) -> String;
}

#[derive(Hash)]
pub(crate) struct Backtrace(String);

impl Backtrace {
    pub fn new_unresolved() -> Self {
        Self(Error::new().stack())
    }

    pub fn format(&mut self) -> std::sync::Arc<str> {
        trim_backtrace(&self.0).into()
    }

    pub fn get_last_relevant_func_name(&mut self) -> std::sync::Arc<str> {
        let trimmed = trim_backtrace(&self.0).into();
        let last_func = get_last_relevant_func_name(&trimmed);
        last_func.into()
    }
}

fn trim_backtrace(mut stack: &str) -> String {
    let start_pattern = "__rust_alloc_zeroed";
    if let Some(start_offset) = stack.find(start_pattern) {
        if let Some(next_newline) = stack[start_offset..].find('\n') {
            stack = &stack[start_offset + next_newline + 1..];
        }
    }

    let end_pattern = "paint_and_schedule"; // normal eframe entry-point
    if let Some(end_offset) = stack.find(end_pattern) {
        if let Some(next_newline) = stack[end_offset..].find('\n') {
            stack = &stack[..end_offset + next_newline];
        }
    }

    stack.split('\n').map(trim_line).collect()
}

/// Example inputs:
/// * `eframe::web::backend::AppRunner::paint::h584aff3234354fd5@http://127.0.0.1:9090/re_viewer.js
///   line 366 > WebAssembly.instantiate:wasm-function[3352]:0x5d46b4`
/// * `getImports/imports.wbg.__wbg_new_83e4891414f9e5c1/<@http://127.0.0.1:9090/re_viewer.js:453:21`
/// * `__rg_realloc@http://127.0.0.1:9090/re_viewer.js line 366 >
///   WebAssembly.instantiate:wasm-function[17996]:0x9b935f`
fn trim_line(mut line: &str) -> String {
    if let Some(index) = line.rfind("::") {
        line = &line[..index];
    }
    if let Some(index) = line.find("/imports.wbg") {
        line = &line[..index];
    }
    if let Some(index) = line.find("@http:") {
        line = &line[..index];
    }
    format!("{line}\n")
}

fn get_last_func_name(backtrace: &String) -> String {
    let stack_lines: Vec<String> = backtrace.split('\n').map(String::from).collect();

    if stack_lines.is_empty() {
        return "[empty backtrace]".to_owned();
    }
    stack_lines.last().unwrap().into()
}

fn get_last_relevant_func_name(backtrace: &String) -> String {
    let stack_lines: Vec<String> = backtrace.split('\n').map(String::from).collect();
    if stack_lines.is_empty() {
        return "[empty backtrace]".to_owned();
    }
    let mut last_func_name = get_last_func_name(backtrace);
    for name in stack_lines.iter() {
        let name_str = name.trim().to_string();
        //if the name starts with alloc it's probably not interesting since it's
        // somewhere in the internals of rust
        if name_str.starts_with("at alloc")
            || name_str.starts_with("at <alloc")
            || name_str.starts_with("at gloss_memory")
            || name_str.starts_with("at<gloss_memory")
            || name_str.starts_with("at std::thread")
            || name_str.starts_with("at <u8 as alloc::vec")
            || name_str.starts_with("at <T as alloc::vec")
            || name_str.starts_with("at __rust_alloc")
            || name_str.starts_with("at __rust_realloc")
            || name_str.starts_with("Error")
            || name_str.starts_with("at imports.wbg")
        {
            continue;
        }

        //found a relevant func
        last_func_name = name_str.to_owned();
        break;
    }
    last_func_name
}
