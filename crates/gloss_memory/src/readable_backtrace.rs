use crate::Backtrace;
use std::sync::Arc;

/// Formatted backtrace.
///
/// Clones without allocating.
#[derive(Clone)]
pub struct ReadableBacktrace {
    /// Human-readable backtrace.
    readable: Arc<str>,
    pub last_relevant_func_name: Arc<str>,
}

impl std::fmt::Display for ReadableBacktrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.readable.fmt(f)
    }
}

impl ReadableBacktrace {
    pub(crate) fn new(mut backtrace: Backtrace) -> Self {
        Self {
            readable: backtrace.format(),
            last_relevant_func_name: backtrace.get_last_relevant_func_name(),
        }
    }
}
unsafe impl Send for ReadableBacktrace {}
unsafe impl Sync for ReadableBacktrace {}
