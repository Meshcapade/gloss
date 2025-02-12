//Mostly from Rerun re_memory crate

use std::sync::Arc;

pub(crate) struct Backtrace(backtrace::Backtrace);

impl Backtrace {
    pub fn new_unresolved() -> Self {
        Self(backtrace::Backtrace::new_unresolved())
    }

    pub fn format(&mut self) -> Arc<str> {
        self.0.resolve();
        let stack = backtrace_to_string(&self.0);
        stack.into()
        // trim_backtrace(&stack).into()
    }

    pub fn get_last_relevant_func_name(&mut self) -> Arc<str> {
        self.0.resolve();
        let last_func = get_last_relevant_func_name(&self.0);
        last_func.into()
    }
}

impl std::hash::Hash for Backtrace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for frame in self.0.frames() {
            frame.ip().hash(state);
        }
    }
}

// fn trim_backtrace(mut stack: &str) -> &str {
//     let start_pattern = "gloss_memory::accounting_allocator::note_alloc\n";
//     if let Some(start_offset) = stack.find(start_pattern) {
//         stack = &stack[start_offset + start_pattern.len()..];
//     }

//     let end_pattern =
// "std::sys_common::backtrace::__rust_begin_short_backtrace";     if let
// Some(end_offset) = stack.find(end_pattern) {         stack =
// &stack[..end_offset];     }

//     stack
// }

// We need to get a `std::fmt::Formatter`, and there is no easy way to do that,
// so we do it the hard way:
struct AnonymizedBacktrace<'a>(&'a backtrace::Backtrace);
impl<'a> std::fmt::Display for AnonymizedBacktrace<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_backtrace_with_fmt(self.0, f)
    }
}

fn backtrace_to_string(backtrace: &backtrace::Backtrace) -> String {
    if backtrace.frames().is_empty() {
        return "[empty backtrace]".to_owned();
    }

    AnonymizedBacktrace(backtrace).to_string()
}

fn format_backtrace_with_fmt(backtrace: &backtrace::Backtrace, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut print_path = |fmt: &mut std::fmt::Formatter<'_>, path: backtrace::BytesOrWideString<'_>| {
        let path = path.into_path_buf();
        let shortened = shorten_source_file_path(&path);
        std::fmt::Display::fmt(&shortened, fmt)
    };

    let style = if fmt.alternate() {
        backtrace::PrintFmt::Full
    } else {
        backtrace::PrintFmt::Short
    };
    let mut f = backtrace::BacktraceFmt::new(fmt, style, &mut print_path);
    f.add_context()?;
    for frame in backtrace.frames() {
        f.frame().backtrace_frame(frame)?;
    }
    f.finish()?;
    Ok(())
}

fn get_last_func_name(backtrace: &backtrace::Backtrace) -> String {
    if backtrace.frames().is_empty() {
        return "[empty backtrace]".to_owned();
    }
    let last_frame = backtrace.frames().last().unwrap();
    let mut last_func_name = "[no_func_found]".to_string();
    for symbol in last_frame.symbols() {
        if let Some(name) = symbol.name() {
            last_func_name = name.to_string();
        }
    }
    last_func_name
}

fn get_last_relevant_func_name(backtrace: &backtrace::Backtrace) -> String {
    if backtrace.frames().is_empty() {
        return "[empty backtrace]".to_owned();
    }
    let mut last_func_name = get_last_func_name(backtrace);
    'outer: for frame in backtrace.frames() {
        for symbol in frame.symbols() {
            if let Some(name) = symbol.name() {
                let name_str = name.to_string().trim().to_string();
                //if the name starts with alloc it's probably not interesting since it's
                // somewhere in the internals of rust
                if name_str.starts_with("alloc")
                    || name_str.starts_with("<alloc")
                    || name_str.starts_with("gloss_memory")
                    || name_str.starts_with("<gloss_memory")
                    || name_str.starts_with("std::thread")
                    || name_str.starts_with("<u8 as alloc::vec")
                    || name_str.starts_with("<T as alloc::vec")
                    || name_str.starts_with("__rust_alloc")
                    || name_str.starts_with("__rust_realloc")
                {
                    continue;
                }

                //found a relevant func
                last_func_name = name_str;
                break 'outer;
            }
        }
    }
    last_func_name
}

/// Anonymize a path to a Rust source file from a callstack.
///
/// Example input:
/// * `crates/rerun/src/main.rs`
/// * `/rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/ops/
///   function.rs`
fn shorten_source_file_path(path: &std::path::Path) -> String {
    // We must make sure we strip everything sensitive (especially user name).
    // The easiest way is to look for `src` and strip everything up to it.

    use itertools::Itertools as _;
    let components = path.iter().map(|path| path.to_string_lossy()).collect_vec();

    // Look for the last `src`:
    if let Some((src_rev_idx, _)) = components.iter().rev().find_position(|&c| c == "src") {
        let src_idx = components.len() - src_rev_idx - 1;
        // Before `src` comes the name of the crate - let's include that:
        let first_index = src_idx.saturating_sub(1);
        components.iter().skip(first_index).format("/").to_string()
    } else {
        // No `src` directory found - weird!
        path.display().to_string()
    }
}
