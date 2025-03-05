//! Documentation for your library.
// #![deny(missing_docs)]

#![doc = include_str!("../../../README.md")]

#[macro_use]
extern crate static_assertions;

use log::log;
use re_memory::{AccountingAllocator, MemoryUse};
use utils_rs::string::float2string;

#[global_allocator]
static GLOBAL: AccountingAllocator<std::alloc::System> = AccountingAllocator::new(std::alloc::System);

pub mod actor;
pub mod camera;
pub mod components;
pub mod config;
pub mod forward_renderer;
pub mod geom;
#[cfg(feature = "with-gui")]
pub mod gui;
pub mod light;
pub mod logger;
pub mod plugin_manager;
pub mod scene;
pub mod viewer;
pub mod viewer_headless;

pub use logger::{gloss_setup_logger, gloss_setup_logger_from_config, gloss_setup_logger_from_config_file};

pub fn print_memory_usage_info(mem_use: MemoryUse, verbosity: log::Level) {
    if let Some(mem_resident) = mem_use.resident {
        log!(verbosity, "MB resident total: {}", (mem_resident / (1024 * 1024)));
    }
    if let Some(mem_counted) = mem_use.counted {
        log!(verbosity, "MB counted total: {}", (mem_counted / (1024 * 1024)));
    }
}

pub fn aa_print_memory_usage_info(show_backtrace: bool, verbosity: log::Level) {
    if let Some(tracks) = re_memory::accounting_allocator::tracking_stats() {
        #[allow(clippy::cast_precision_loss)]
        for cb in tracks.top_callstacks.iter() {
            let mb_cb = cb.extant.size as f32 / (1024.0 * 1024.0);
            if show_backtrace {
                log!(verbosity, "MB: {} Callstack: {}", float2string(mb_cb, 1), cb.readable_backtrace);
            } else {
                log!(verbosity, "MB: {} Func: {}", float2string(mb_cb, 1), cb.readable_backtrace,);
            }
        }
    }
}

fn set_panic_hook() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(move |info| {
                web_sys::console::error_1(&format!("PANICKED: Will print memory usage info:").into());
                // MemoryUse::capture().print_memory_usage_info(log::Level::Error);
                // accounting_allocator::print_memory_usage_info(false, log::Level::Error);
                print_memory_usage_info(MemoryUse::capture(), log::Level::Error);
                aa_print_memory_usage_info(false, log::Level::Error);
                console_error_panic_hook::hook(info);
            }));
        }else{
            let default_panic = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                // println!("PANICKED: Will print memory usage info:");
                // MemoryUse::capture().print_memory_usage_info(log::Level::Error);
                // let enabled_backtrace = std::env::var("RUST_BACKTRACE").map_or(false, |_| true);
                // accounting_allocator::print_memory_usage_info(enabled_backtrace, log::Level::Error);
                default_panic(info);
            }));
        }
    }
}
