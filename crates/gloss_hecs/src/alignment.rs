//COPIED FROM use std::ptr::Alignment;

use std::convert::{TryFrom, TryInto};
// use crate::intrinsics::assert_unsafe_precondition;
use std::{cmp, fmt, hash, mem, num::NonZeroUsize};

#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

/// A type storing a `usize` which is a power of two, and thus
/// represents a possible alignment in the rust abstract machine.
///
/// Note that particularly large alignments, while representable in this type,
/// are likely not to be supported by actual allocators and linkers.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Alignment(AlignmentEnum);

// Alignment is `repr(usize)`, but via extra steps.
const _: () = assert!(mem::size_of::<Alignment>() == mem::size_of::<usize>());
const _: () = assert!(mem::align_of::<Alignment>() == mem::align_of::<usize>());

fn _alignment_can_be_structurally_matched(a: Alignment) -> bool {
    matches!(a, Alignment::MIN)
}

impl Alignment {
    /// Minimum alignment value
    pub const MIN: Self = Self(AlignmentEnum::_Align1Shl0);

    /// Returns the alignment for a type.
    ///
    /// This provides the same numerical value as [`mem::align_of`],
    /// but in an `Alignment` instead of a `usize`.
    #[inline]
    pub const fn of<T>() -> Self {
        // SAFETY: rustc ensures that type alignment is always a power of two.
        unsafe { Alignment::new_unchecked(mem::align_of::<T>()) }
    }

    /// Creates an `Alignment` from a `usize`, or returns `None` if it's
    /// not a power of two.
    ///
    /// Note that `0` is not a power of two, nor a valid alignment.
    #[inline]
    pub const fn new(align: usize) -> Option<Self> {
        if align.is_power_of_two() {
            // SAFETY: Just checked it only has one bit set
            Some(unsafe { Self::new_unchecked(align) })
        } else {
            None
        }
    }

    /// Creates an `Alignment` from a power-of-two `usize`.
    ///
    /// # Safety
    ///
    /// `align` must be a power of two.
    ///
    /// Equivalently, it must be `1 << exp` for some `exp` in `0..usize::BITS`.
    /// It must *not* be zero.
    #[inline]
    pub const unsafe fn new_unchecked(align: usize) -> Self {
        // // SAFETY: Precondition passed to the caller.
        // unsafe {
        //     assert_unsafe_precondition!(
        //        "Alignment::new_unchecked requires a power of two",
        //         (align: usize) => align.is_power_of_two()
        //     )
        // };

        // SAFETY: By precondition, this must be a power of two, and
        // our variants encompass all possible powers of two.
        unsafe { mem::transmute::<usize, Alignment>(align) }
    }

    /// Returns the alignment as a [`usize`]
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns the alignment as a [`NonZeroUsize`]
    #[inline]
    pub const fn as_nonzero(self) -> NonZeroUsize {
        // SAFETY: All the discriminants are non-zero.
        unsafe { NonZeroUsize::new_unchecked(self.as_usize()) }
    }

    /// log2
    #[inline]
    pub fn log2(self) -> u32 {
        self.as_nonzero().trailing_zeros()
    }
}

impl fmt::Debug for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (1 << {:?})", self.as_nonzero(), self.log2())
    }
}

impl TryFrom<NonZeroUsize> for Alignment {
    // type Error = num::TryFromIntError;
    type Error = ();

    #[inline]
    fn try_from(align: NonZeroUsize) -> Result<Alignment, Self::Error> {
        align.get().try_into()
    }
}

impl TryFrom<usize> for Alignment {
    // type Error = num::TryFromIntError;
    type Error = ();

    #[inline]
    fn try_from(align: usize) -> Result<Alignment, Self::Error> {
        Self::new(align).ok_or(())
    }
}

impl From<Alignment> for NonZeroUsize {
    #[inline]
    fn from(align: Alignment) -> NonZeroUsize {
        align.as_nonzero()
    }
}

impl From<Alignment> for usize {
    #[inline]
    fn from(align: Alignment) -> usize {
        align.as_usize()
    }
}

impl cmp::Ord for Alignment {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_nonzero().get().cmp(&other.as_nonzero().get())
    }
}

impl cmp::PartialOrd for Alignment {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl hash::Hash for Alignment {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_nonzero().hash(state);
    }
}

#[cfg(target_pointer_width = "16")]
type AlignmentEnum = AlignmentEnum16;
#[cfg(target_pointer_width = "32")]
type AlignmentEnum = AlignmentEnum32;
#[cfg(target_pointer_width = "64")]
type AlignmentEnum = AlignmentEnum64;

#[allow(dead_code)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u16)]
enum AlignmentEnum16 {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
}

#[allow(dead_code)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
enum AlignmentEnum32 {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
}

#[allow(dead_code)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
enum AlignmentEnum64 {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
    _Align1Shl32 = 1 << 32,
    _Align1Shl33 = 1 << 33,
    _Align1Shl34 = 1 << 34,
    _Align1Shl35 = 1 << 35,
    _Align1Shl36 = 1 << 36,
    _Align1Shl37 = 1 << 37,
    _Align1Shl38 = 1 << 38,
    _Align1Shl39 = 1 << 39,
    _Align1Shl40 = 1 << 40,
    _Align1Shl41 = 1 << 41,
    _Align1Shl42 = 1 << 42,
    _Align1Shl43 = 1 << 43,
    _Align1Shl44 = 1 << 44,
    _Align1Shl45 = 1 << 45,
    _Align1Shl46 = 1 << 46,
    _Align1Shl47 = 1 << 47,
    _Align1Shl48 = 1 << 48,
    _Align1Shl49 = 1 << 49,
    _Align1Shl50 = 1 << 50,
    _Align1Shl51 = 1 << 51,
    _Align1Shl52 = 1 << 52,
    _Align1Shl53 = 1 << 53,
    _Align1Shl54 = 1 << 54,
    _Align1Shl55 = 1 << 55,
    _Align1Shl56 = 1 << 56,
    _Align1Shl57 = 1 << 57,
    _Align1Shl58 = 1 << 58,
    _Align1Shl59 = 1 << 59,
    _Align1Shl60 = 1 << 60,
    _Align1Shl61 = 1 << 61,
    _Align1Shl62 = 1 << 62,
    _Align1Shl63 = 1 << 63,
}
