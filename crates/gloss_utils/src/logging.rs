// mostly copied from here:
// https://github.com/seanmonstar/pretty-env-logger/blob/master/src/lib.rs
// and from: https://github.com/LuckyTurtleDev/my-env-logger-style/blob/main/src/lib.rs

use env_logger::fmt::Formatter;
use log::{Level, Record};
// #[cfg(feature = "custom-arg-formatter")]
// use once_cell::sync::OnceCell;
#[cfg(feature = "log_with_time")]
use std::sync::atomic::AtomicU8;
use std::{
    io,
    io::Write,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

static MAX_MODULE_LEN: AtomicUsize = AtomicUsize::new(0);
static SHOW_MODULE: AtomicBool = AtomicBool::new(true);
static SHOW_EMOJIS: AtomicBool = AtomicBool::new(true);
#[cfg(feature = "log_with_time")]
static SHOW_TIME: AtomicU8 = AtomicU8::new(env_logger::TimestampPrecision::Seconds as u8);
// #[cfg(feature = "custom-arg-formatter")]
// static ARG_FORMATTER: OnceCell<Box<dyn ArgFormatter + Send + Sync>> =
// OnceCell::new();

pub use env_logger;

/// return the current module len and set the module length to the maximum of
/// the current value and the given `len`.
///
/// Usefull if you already know the length of module and would like to have an
/// consistant indentation from the beginnig.
pub fn get_set_max_module_len(len: usize) -> usize {
    let module_len = MAX_MODULE_LEN.load(Ordering::Relaxed);
    if module_len < len {
        MAX_MODULE_LEN.store(len, Ordering::Relaxed);
    }
    module_len
}

///log formater witch can be used at the
/// [`format()`](env_logger::Builder::format()) function of the
/// [`env_logger::Builder`].
#[allow(clippy::missing_errors_doc)]
pub fn format(buf: &mut Formatter, record: &Record<'_>) -> io::Result<()> {
    let mut bold = buf.style();
    bold.set_bold(true);
    let mut dimmed = buf.style();
    dimmed.set_dimmed(true);

    #[cfg(feature = "log_with_time")]
    {
        let show_time = SHOW_TIME.load(Ordering::Relaxed);
        // safety: SHOW_TIME is inilized with TimestampPrecision::Seconds
        // and can only be written by using set_timestamp_precision()
        match unsafe { std::mem::transmute::<u8, env_logger::TimestampPrecision>(show_time) } {
            env_logger::TimestampPrecision::Seconds => {
                write!(buf, "{} ", dimmed.value(buf.timestamp_seconds()))
            }
            env_logger::TimestampPrecision::Millis => {
                write!(buf, "{} ", dimmed.value(buf.timestamp_millis()))
            }
            env_logger::TimestampPrecision::Micros => {
                write!(buf, "{} ", dimmed.value(buf.timestamp_micros()))
            }
            env_logger::TimestampPrecision::Nanos => {
                write!(buf, "{} ", dimmed.value(buf.timestamp_nanos()))
            }
        }?;
    }

    let level_style = buf.default_level_style(record.level());
    let level_symbol = if SHOW_EMOJIS.load(Ordering::Relaxed) {
        match record.level() {
            //💥 and 🔬 are 2 chars big at the terminal. How does it look with other fonts/terminals?
            Level::Trace => "🔬",
            Level::Debug => " ⚙️",
            Level::Info => " ℹ",
            Level::Warn => " ⚠",
            Level::Error => "💥",
        }
    } else {
        ""
    };
    write!(buf, "{level_symbol} {:5} ", level_style.value(record.level()))?;

    if SHOW_MODULE.load(Ordering::Relaxed) {
        let module = record.module_path().unwrap_or_default();
        let module_with_line_nr = module.to_owned() + "::" + &record.line().unwrap_or(0).to_string();
        let module_len = get_set_max_module_len(module_with_line_nr.len() + 2); //+1 because we add a "::" between module path and line_nr
        write!(buf, "{:module_len$} {} ", dimmed.value(module_with_line_nr), bold.value('>'))?;
    }

    // #[cfg(feature = "custom-arg-formatter")]
    // if let Some(formatter) = ARG_FORMATTER.get() {
    // return formatter.arg_format(buf, record);
    // }
    writeln!(buf, "{}", record.args())
}
