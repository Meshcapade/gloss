//Mostly from Rerun re_memory crate
//! Track allocations and memory use.

use std::sync::atomic::{
    AtomicBool, AtomicUsize,
    Ordering::{self, Relaxed},
};

// use crate::{Backtrace, BacktraceHash};
// use atomic::Atomic;
#[cfg(not(target_arch = "wasm32"))]
use log::info;

use log::log;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use utils_rs::string::float2string;

use crate::{
    allocation_tracker::{AllocationTracker, CallstackStatistics, PtrHash},
    CountAndSize,
};

/// Only track allocations of at least this size.
const SMALL_SIZE: usize = 128; // TODO(emilk): make this setable by users

/// Allocations smaller than are stochastically sampled.
const MEDIUM_SIZE: usize = 8 * 1024; // TODO(emilk): make this setable by users

// TODO(emilk): yet another tier would maybe make sense, with a different
// stochastic rate.

/// Statistics about extant allocations larger than [`MEDIUM_SIZE`].
static BIG_ALLOCATION_TRACKER: Lazy<Mutex<AllocationTracker>> = Lazy::new(|| Mutex::new(AllocationTracker::with_stochastic_rate(1)));

/// Statistics about some extant allocations larger than  [`SMALL_SIZE`] but
/// smaller than [`MEDIUM_SIZE`].
static MEDIUM_ALLOCATION_TRACKER: Lazy<Mutex<AllocationTracker>> = Lazy::new(|| Mutex::new(AllocationTracker::with_stochastic_rate(64)));

thread_local! {
    /// Used to prevent re-entrancy when tracking allocations.
    ///
    /// Tracking an allocation (taking its backtrace etc) can itself create allocations.
    /// We don't want to track those allocations, or we will have infinite recursion.
    static IS_THREAD_IN_ALLOCATION_TRACKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// ----------------------------------------------------------------------------

struct AtomicCountAndSize {
    /// Number of allocations.
    pub count: AtomicUsize,

    /// Number of bytes.
    pub size: AtomicUsize,
}

impl AtomicCountAndSize {
    pub const fn zero() -> Self {
        Self {
            count: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
        }
    }

    fn load(&self) -> CountAndSize {
        CountAndSize {
            count: self.count.load(Relaxed),
            size: self.size.load(Relaxed),
        }
    }

    /// Add an allocation.
    fn add(&self, size: usize) {
        self.count.fetch_add(1, Relaxed);
        self.size.fetch_add(size, Relaxed);
    }

    /// Remove an allocation.
    fn sub(&self, size: usize) {
        self.count.fetch_sub(1, Relaxed);
        self.size.fetch_sub(size, Relaxed);
    }
}

struct GlobalStats {
    /// All extant allocations.
    pub live: AtomicCountAndSize,

    /// Do detailed statistics of allocations?
    /// This is expensive, but sometimes useful!
    pub track_callstacks: AtomicBool,

    /// The live allocations not tracked by any [`AllocationTracker`].
    pub untracked: AtomicCountAndSize,

    /// All live allocations sampled by the stochastic medium
    /// [`AllocationTracker`].
    pub stochastically_tracked: AtomicCountAndSize,

    /// All live allocations tracked by the large [`AllocationTracker`].
    pub fully_tracked: AtomicCountAndSize,

    /// The live allocations done by [`AllocationTracker`] used for internal
    /// book-keeping.
    pub overhead: AtomicCountAndSize,

    /// Pointer to the first block of memory allocated ("first" defined as the
    /// ptr with the minimal value) This is useful especially for WASM where
    /// there is a linear memory model so the full memory allocated can be
    /// obtained from `max_ptr` - `min_ptr`
    pub min_ptr: AtomicUsize,
    // pub min_alloc_callstack: Atomic<Option<ReadableBacktrace>>,
    /// Pointer to the last block of memory allocated ("last" defined as the ptr
    /// with the maximal value) This is useful especially for WASM where there
    /// is a linear memory model so the full memory allocated can be obtained
    /// from `max_ptr` - `min_ptr`
    pub max_ptr: AtomicUsize,
    // pub max_alloc_callstack: Atomic<Option<ReadableBacktrace>>,
}

// ----------------------------------------------------------------------------

static GLOBAL_STATS: GlobalStats = GlobalStats {
    live: AtomicCountAndSize::zero(),
    track_callstacks: AtomicBool::new(false),
    untracked: AtomicCountAndSize::zero(),
    stochastically_tracked: AtomicCountAndSize::zero(),
    fully_tracked: AtomicCountAndSize::zero(),
    overhead: AtomicCountAndSize::zero(),
    min_ptr: AtomicUsize::new(usize::MAX),
    max_ptr: AtomicUsize::new(0),
    // min_alloc_callstack: Atomic::new(None),
    // max_alloc_callstack: Atomic::new(None),
};

/// Total number of live allocations,
/// and the number of live bytes allocated as tracked by
/// [`AccountingAllocator`].
///
/// Returns `None` if [`AccountingAllocator`] is not used.
pub fn global_allocs() -> Option<CountAndSize> {
    let count_and_size = GLOBAL_STATS.live.load();
    (count_and_size.count > 0).then_some(count_and_size)
}

/// Are we doing (slightly expensive) tracking of the callstacks of large
/// allocations?
pub fn is_tracking_callstacks() -> bool {
    GLOBAL_STATS.track_callstacks.load(Relaxed)
}

/// Should we do (slightly expensive) tracking of the callstacks of large
/// allocations?
///
/// See also [`turn_on_tracking_if_env_var`].
///
/// Requires that you have installed the [`AccountingAllocator`].
pub fn set_tracking_callstacks(track: bool) {
    GLOBAL_STATS.track_callstacks.store(track, Relaxed);
}

/// WASM uses a linear memory model so the whole memory can be approximated by
/// the min and max pointer to the memory
#[cfg(target_arch = "wasm32")]
pub fn bytes_allocated_approx() -> usize {
    let min = GLOBAL_STATS.min_ptr.load(Relaxed);
    let max = GLOBAL_STATS.max_ptr.load(Relaxed);
    max - min
}

/// Turn on callstack tracking (slightly expensive) if a given env-var is set.
///
/// See also [`set_tracking_callstacks`].
///
/// Requires that you have installed the [`AccountingAllocator`].
#[cfg(not(target_arch = "wasm32"))]
pub fn turn_on_tracking_if_env_var(env_var: &str) {
    if std::env::var(env_var).is_ok() {
        set_tracking_callstacks(true);
        info!("{env_var} found - turning on tracking of all large allocations");
    }
}

// ----------------------------------------------------------------------------

const MAX_CALLSTACKS: usize = 128;

pub struct TrackingStatistics {
    /// Allocations smaller than these are left untracked.
    pub track_size_threshold: usize,

    /// All live allocations that we are NOT tracking (because they were below
    /// [`Self::track_size_threshold`]).
    pub untracked: CountAndSize,

    /// All live allocations sampled of medium size, stochastically sampled.
    pub stochastically_tracked: CountAndSize,

    /// All live largish allocations, fully tracked.
    pub fully_tracked: CountAndSize,

    /// All live allocations used for internal book-keeping.
    pub overhead: CountAndSize,

    /// The most popular callstacks.
    pub top_callstacks: Vec<CallstackStatistics>,
}

/// Gather statistics from the live tracking, if enabled.
///
/// Enable this with [`set_tracking_callstacks`], preferably the first thing you
/// do in `main`.
///
/// Requires that you have installed the [`AccountingAllocator`].
pub fn tracking_stats() -> Option<TrackingStatistics> {
    /// NOTE: we use a rather large [`smallvec::SmallVec`] here to avoid dynamic
    /// allocations, which would otherwise confuse the memory tracking.
    fn tracker_stats(allocation_tracker: &AllocationTracker) -> smallvec::SmallVec<[CallstackStatistics; MAX_CALLSTACKS]> {
        let top_callstacks: smallvec::SmallVec<[CallstackStatistics; MAX_CALLSTACKS]> =
            allocation_tracker.top_callstacks(MAX_CALLSTACKS).into_iter().collect();
        assert!(!top_callstacks.spilled(), "We shouldn't leak any allocations");
        top_callstacks
    }

    // GLOBAL_STATS.track_callstacks.load(Relaxed).then(|| {
    let stats = IS_THREAD_IN_ALLOCATION_TRACKER.with(|is_thread_in_allocation_tracker| {
        // prevent double-lock of ALLOCATION_TRACKER:
        is_thread_in_allocation_tracker.set(true);
        let mut top_big_callstacks = tracker_stats(&BIG_ALLOCATION_TRACKER.lock());
        let mut top_medium_callstacks = tracker_stats(&MEDIUM_ALLOCATION_TRACKER.lock());
        is_thread_in_allocation_tracker.set(false);

        let mut top_callstacks: Vec<_> = top_big_callstacks.drain(..).chain(top_medium_callstacks.drain(..)).collect();
        top_callstacks.sort_by_key(|c| std::cmp::Reverse(c.extant.size));

        TrackingStatistics {
            track_size_threshold: SMALL_SIZE,
            untracked: GLOBAL_STATS.untracked.load(),
            stochastically_tracked: GLOBAL_STATS.stochastically_tracked.load(),
            fully_tracked: GLOBAL_STATS.fully_tracked.load(),
            overhead: GLOBAL_STATS.overhead.load(),
            top_callstacks,
        }
    });

    Some(stats)
    // })
}

pub fn print_memory_usage_info(show_backtrace: bool, verbosity: log::Level) {
    if let Some(tracks) = tracking_stats() {
        #[allow(clippy::cast_precision_loss)]
        for cb in tracks.top_callstacks.iter() {
            let mb_cb = cb.extant.size as f32 / (1024.0 * 1024.0);
            if show_backtrace {
                log!(verbosity, "MB: {} Callstack: {}", float2string(mb_cb, 1), cb.readable_backtrace);
            } else {
                log!(
                    verbosity,
                    "MB: {} Func: {}",
                    float2string(mb_cb, 1),
                    cb.readable_backtrace.last_relevant_func_name
                );
            }
        }
    }
}

pub fn live_allocs_list() -> Vec<(usize, usize)> {
    /// NOTE: we use a rather large [`smallvec::SmallVec`] here to avoid dynamic
    /// allocations, which would otherwise confuse the memory tracking.
    fn get_allocs(allocation_tracker: &AllocationTracker) -> smallvec::SmallVec<[(usize, usize); MAX_CALLSTACKS]> {
        let top_callstacks: smallvec::SmallVec<[(usize, usize); MAX_CALLSTACKS]> =
            allocation_tracker.top_live_allocs(MAX_CALLSTACKS).into_iter().collect();
        assert!(!top_callstacks.spilled(), "We shouldn't leak any allocations");
        top_callstacks
    }
    let live_allocs_big = get_allocs(&BIG_ALLOCATION_TRACKER.lock());
    // let mut live_allocs_medium =
    // MEDIUM_ALLOCATION_TRACKER.lock().live_allocs_list();

    // let live_allocs: Vec<_> = live_allocs_big
    //     .drain(..)
    //     // .chain(live_allocs_medium.drain(..))
    //     .collect();

    // live_allocs
    live_allocs_big.to_vec()
}

pub fn min_ptr_alloc_memory() -> usize {
    GLOBAL_STATS.min_ptr.load(Relaxed)
}

// ----------------------------------------------------------------------------

/// Install this as the global allocator to get memory usage tracking.
///
/// Use [`set_tracking_callstacks`] or [`turn_on_tracking_if_env_var`] to turn
/// on memory tracking. Collect the stats with [`tracking_stats`].
///
/// Usage:
/// ```
/// use gloss_memory::AccountingAllocator;
///
/// #[global_allocator]
/// static GLOBAL: AccountingAllocator<std::alloc::System> =
///     AccountingAllocator::new(std::alloc::System);
/// ```
#[derive(Default)]
pub struct AccountingAllocator<InnerAllocator> {
    allocator: InnerAllocator,
}

impl<InnerAllocator> AccountingAllocator<InnerAllocator> {
    pub const fn new(allocator: InnerAllocator) -> Self {
        Self { allocator }
    }
}

#[allow(unsafe_code)]
// SAFETY:
// We just do book-keeping and then let another allocator do all the actual
// work.
unsafe impl<InnerAllocator: std::alloc::GlobalAlloc> std::alloc::GlobalAlloc for AccountingAllocator<InnerAllocator> {
    #[allow(clippy::let_and_return)]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY:
        // We just do book-keeping and then let another allocator do all the actual
        // work.
        let ptr = unsafe { self.allocator.alloc(layout) };

        note_alloc(ptr, layout.size());

        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY:
        // We just do book-keeping and then let another allocator do all the actual
        // work.
        let ptr = unsafe { self.allocator.alloc_zeroed(layout) };

        note_alloc(ptr, layout.size());

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY:
        // We just do book-keeping and then let another allocator do all the actual
        // work.
        unsafe { self.allocator.dealloc(ptr, layout) };

        note_dealloc(ptr, layout.size());
    }

    unsafe fn realloc(&self, old_ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        note_dealloc(old_ptr, layout.size());

        // SAFETY:
        // We just do book-keeping and then let another allocator do all the actual
        // work.
        let new_ptr = unsafe { self.allocator.realloc(old_ptr, layout, new_size) };

        note_alloc(new_ptr, new_size);

        new_ptr
    }
}

#[inline]
fn note_alloc(ptr: *mut u8, size: usize) {
    GLOBAL_STATS.live.add(size);

    //WASM can only allocate memory so we use the min and max to keep track of the
    // maximum memory we allocated
    let ptr_val = ptr as usize;
    let min_ptr_val = ptr as usize;
    let max_ptr_val = ptr as usize + size;
    GLOBAL_STATS.min_ptr.fetch_min(min_ptr_val, Ordering::Relaxed);
    GLOBAL_STATS.max_ptr.fetch_max(max_ptr_val, Ordering::Relaxed);

    //store also a backtrace
    // let unresolved_backtrace = Backtrace::new_unresolved();
    // let backtrace = ReadableBacktrace::new(unresolved_backtrace);

    // if min_ptr_val != prev_min {
    //     // GLOBAL_STATS.min_alloc_callstack = Some(backtrace);
    //     // let txt = "min_backtrace ".to_owned() + &backtrace.to_string();
    //     println!("min_backtrace {}", backtrace);
    // }
    // if max_ptr_val != prev_max {
    //     println!("max_backtrace {}", backtrace);
    // }

    if GLOBAL_STATS.track_callstacks.load(Relaxed) {
        if size < SMALL_SIZE {
            // Too small to track.
            GLOBAL_STATS.untracked.add(size);
        } else {
            // Big enough to track - but make sure we don't create a deadlock by trying to
            // track the allocations made by the allocation tracker:

            IS_THREAD_IN_ALLOCATION_TRACKER.with(|is_thread_in_allocation_tracker| {
                if is_thread_in_allocation_tracker.get() {
                    // This is the ALLOCATION_TRACKER allocating memory.
                    GLOBAL_STATS.overhead.add(size);
                } else {
                    is_thread_in_allocation_tracker.set(true);

                    let ptr_hash = PtrHash::new(ptr);
                    if size < MEDIUM_SIZE {
                        GLOBAL_STATS.stochastically_tracked.add(size);
                        MEDIUM_ALLOCATION_TRACKER.lock().on_alloc(ptr_hash, ptr_val, size);
                    } else {
                        GLOBAL_STATS.fully_tracked.add(size);
                        BIG_ALLOCATION_TRACKER.lock().on_alloc(ptr_hash, ptr_val, size);
                    }

                    is_thread_in_allocation_tracker.set(false);
                }
            });
        }
    }
}

#[inline]
fn note_dealloc(ptr: *mut u8, size: usize) {
    GLOBAL_STATS.live.sub(size);

    let ptr_val = ptr as usize;

    if GLOBAL_STATS.track_callstacks.load(Relaxed) {
        if size < SMALL_SIZE {
            // Too small to track.
            GLOBAL_STATS.untracked.sub(size);
        } else {
            // Big enough to track - but make sure we don't create a deadlock by trying to
            // track the allocations made by the allocation tracker:
            IS_THREAD_IN_ALLOCATION_TRACKER.with(|is_thread_in_allocation_tracker| {
                if is_thread_in_allocation_tracker.get() {
                    // This is the ALLOCATION_TRACKER freeing memory.
                    GLOBAL_STATS.overhead.sub(size);
                } else {
                    is_thread_in_allocation_tracker.set(true);

                    let ptr_hash = PtrHash::new(ptr);
                    if size < MEDIUM_SIZE {
                        GLOBAL_STATS.stochastically_tracked.sub(size);
                        MEDIUM_ALLOCATION_TRACKER.lock().on_dealloc(ptr_hash, ptr_val, size);
                    } else {
                        GLOBAL_STATS.fully_tracked.sub(size);
                        BIG_ALLOCATION_TRACKER.lock().on_dealloc(ptr_hash, ptr_val, size);
                    }

                    is_thread_in_allocation_tracker.set(false);
                }
            });
        }
    }
}
