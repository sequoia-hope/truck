use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// ID structure with `Copy`, `Hash` and `Eq` using raw pointers
pub struct ID<T>(usize, PhantomData<T>);

impl<T> ID<T> {
    /// Creates the ID by a raw pointer.
    #[inline(always)]
    pub fn new(ptr: *const T) -> ID<T> {
        ID(ptr as usize, PhantomData)
    }
}

impl<T> Clone for ID<T> {
    #[inline(always)]
    fn clone(&self) -> ID<T> {
        *self
    }
}

impl<T> Copy for ID<T> {}

impl<T> Hash for ID<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T> PartialEq for ID<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for ID<T> {}

impl<T> Ord for ID<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> PartialOrd for ID<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Debug for ID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.write_fmt(format_args!("0x{:x}", self.0))
    }
}

/// Deterministic identity for boolean operations.
///
/// Unlike pointer-derived `ID<T>`, `DetId` values are sequential integers
/// issued by a `DetContext`. Two runs of the same boolean operation produce
/// identical `DetId` sequences, making iteration order over `BTreeMap<DetId, _>`
/// fully deterministic across runs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DetId(u64);

impl DetId {
    /// Create a `DetId` from a raw counter value (for testing).
    #[inline]
    pub fn from_raw(val: u64) -> Self {
        DetId(val)
    }

    /// Get the raw counter value.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Debug for DetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "det#{}", self.0)
    }
}

/// Context for issuing deterministic IDs during a boolean operation.
///
/// Create one `DetContext` at the start of a boolean operation and use
/// `next_id()` to assign sequential IDs to entities. Fresh contexts
/// always start from 0, so two runs produce the same ID sequence.
#[derive(Debug)]
pub struct DetContext {
    counter: std::sync::atomic::AtomicU64,
}

impl DetContext {
    /// Create a new context starting from ID 0.
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Issue the next deterministic ID.
    pub fn next_id(&self) -> DetId {
        DetId(
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }
}

impl Default for DetContext {
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn debug_backward_compatibility() {
    let x: f64 = 3.0;
    let id = ID::new(&x);
    let a = format!("{id:?}");
    let b = format!("{:p}", &x);
    assert_eq!(a, b);
}

#[test]
fn detid_monotonic() {
    let ctx = DetContext::new();
    let id1 = ctx.next_id();
    let id2 = ctx.next_id();
    let id3 = ctx.next_id();
    assert!(id1 < id2);
    assert!(id2 < id3);
    assert_eq!(id1.raw(), 0);
    assert_eq!(id2.raw(), 1);
    assert_eq!(id3.raw(), 2);
}

#[test]
fn detid_deterministic_across_calls() {
    let ctx1 = DetContext::new();
    let ids1: Vec<_> = (0..100).map(|_| ctx1.next_id()).collect();
    let ctx2 = DetContext::new();
    let ids2: Vec<_> = (0..100).map(|_| ctx2.next_id()).collect();
    assert_eq!(ids1, ids2, "Same sequence of IDs from fresh contexts");
}

#[test]
fn detid_ord_matches_creation_order() {
    let ctx = DetContext::new();
    let mut ids: Vec<DetId> = (0..50).map(|_| ctx.next_id()).collect();
    let original = ids.clone();
    ids.sort();
    assert_eq!(ids, original, "Sorted order should match creation order");
}
