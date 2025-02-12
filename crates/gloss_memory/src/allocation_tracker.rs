//Mostly from Rerun re_memory crate

use itertools::Itertools;

use crate::{Backtrace, BacktraceHash, ReadableBacktrace};

use crate::CountAndSize;

// ----------------------------------------------------------------------------

/// A hash of a pointer address.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PtrHash(u64);

impl nohash_hasher::IsEnabled for PtrHash {}

impl PtrHash {
    #[inline]
    pub fn new(ptr: *mut u8) -> Self {
        let hash = ahash::RandomState::with_seeds(1, 2, 3, 4).hash_one(ptr);
        Self(hash)
    }
}

// ----------------------------------------------------------------------------

/// Per-callstack statistics.
#[derive(Clone)]
pub struct CallstackStatistics {
    /// For when we print this statistic.
    pub readable_backtrace: ReadableBacktrace,

    /// If this was stochastically sampled - at what rate?
    ///
    /// A `stochastic_rate` of `10` means that we only sampled 1 in 10
    /// allocations.
    ///
    /// (so this is actually an interval rather than rate…).
    pub stochastic_rate: usize,

    /// Live allocations at this callstack.
    ///
    /// You should multiply this by [`Self::stochastic_rate`] to get an estimate
    /// of the real data.
    pub extant: CountAndSize,
}

// ----------------------------------------------------------------------------

/// Track the callstacks of allocations.
pub struct AllocationTracker {
    /// Sample every N allocations. Must be power-of-two.
    stochastic_rate: usize,

    /// De-duplicated readable backtraces.
    readable_backtraces: nohash_hasher::IntMap<BacktraceHash, ReadableBacktrace>,

    /// Current live allocations (`ptr_hash`, backtracehash, ptr, size)
    live_allocs: ahash::HashMap<PtrHash, (BacktraceHash, usize, usize)>,
    // live_allocs: ahash::HashMap<PtrHash, BacktraceHash>,
    /// How much memory is allocated by each callstack?
    callstack_stats: nohash_hasher::IntMap<BacktraceHash, CountAndSize>,
}

impl AllocationTracker {
    pub fn with_stochastic_rate(stochastic_rate: usize) -> Self {
        assert!(stochastic_rate != 0);
        assert!(stochastic_rate.is_power_of_two());
        Self {
            stochastic_rate,
            readable_backtraces: nohash_hasher::IntMap::default(),
            live_allocs: ahash::HashMap::default(),
            callstack_stats: nohash_hasher::IntMap::default(),
        }
    }

    fn should_sample(&self, ptr: PtrHash) -> bool {
        ptr.0 & (self.stochastic_rate as u64 - 1) == 0
    }

    pub fn on_alloc(&mut self, ptr_hash: PtrHash, ptr: usize, size: usize) {
        if !self.should_sample(ptr_hash) {
            return;
        }

        let unresolved_backtrace = Backtrace::new_unresolved();
        let hash = BacktraceHash::new(&unresolved_backtrace);

        self.readable_backtraces
            .entry(hash)
            .or_insert_with(|| ReadableBacktrace::new(unresolved_backtrace));

        {
            self.callstack_stats.entry(hash).or_default().add(size);
        }

        self.live_allocs.insert(ptr_hash, (hash, ptr, size));
    }

    pub fn on_dealloc(&mut self, ptr_hash: PtrHash, _ptr: usize, size: usize) {
        if !self.should_sample(ptr_hash) {
            return;
        }

        if let Some((hash, _ptr, _size)) = self.live_allocs.remove(&ptr_hash) {
            if let std::collections::hash_map::Entry::Occupied(mut entry) = self.callstack_stats.entry(hash) {
                let stats = entry.get_mut();
                stats.sub(size);

                // Free up some memory:
                if stats.size == 0 {
                    entry.remove();
                }
            }
        }
    }

    /// Return the `n` callstacks that currently is using the most memory.
    pub fn top_callstacks(&self, n: usize) -> Vec<CallstackStatistics> {
        let mut vec: Vec<_> = self
            .callstack_stats
            .iter()
            .filter(|(_hash, c)| c.count > 0)
            .filter_map(|(hash, c)| {
                Some(CallstackStatistics {
                    readable_backtrace: self.readable_backtraces.get(hash)?.clone(),
                    stochastic_rate: self.stochastic_rate,
                    extant: *c,
                })
            })
            .collect();
        vec.sort_by_key(|stats| std::cmp::Reverse(stats.extant.size));
        vec.truncate(n);
        vec.shrink_to_fit();
        vec
    }

    pub fn top_live_allocs(&self, n: usize) -> Vec<(usize, usize)> {
        let mut vec = self.live_allocs.values().map(|(_v_bt, v_ptr, v_size)| (*v_ptr, *v_size)).collect_vec();
        vec.sort_by_key(|(_ptr, size)| std::cmp::Reverse(*size));
        vec.truncate(n);
        vec.shrink_to_fit();
        vec
    }
}
