use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

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
    counter: AtomicU64,
}

impl DetContext {
    /// Create a new context starting from ID 0.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Issue the next deterministic ID.
    pub fn next_id(&self) -> DetId {
        DetId(
            self.counter
                .fetch_add(1, Ordering::Relaxed),
        )
    }
}

impl Default for DetContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SequentialID — deterministic topology entity identity
// ---------------------------------------------------------------------------

/// Global monotonic counter for topology entity IDs.
///
/// Single-threaded WASM + sequential Rust guarantees that creation order is
/// deterministic across runs. Counter starts at 1 (0 is reserved as "unset").
static NEXT_SEQ_ID: AtomicU64 = AtomicU64::new(1);

/// Allocate the next sequential ID. Called by topology constructors.
#[inline]
pub fn next_sequential_id() -> u64 {
    NEXT_SEQ_ID.fetch_add(1, Ordering::Relaxed)
}

/// Reset the global sequence counter to 1. **Test-only.**
///
/// # Safety (logical)
/// Must only be called when no topology entities from a previous sequence
/// are still alive — otherwise two entities could share the same ID.
pub fn reset_id_sequence() {
    NEXT_SEQ_ID.store(1, Ordering::Relaxed);
}

/// Return the current counter value without incrementing (for diagnostics).
#[inline]
pub fn current_sequence_value() -> u64 {
    NEXT_SEQ_ID.load(Ordering::Relaxed)
}

/// Deterministic, creation-order identity for topology entities.
///
/// Replaces pointer-derived `ID<T>` for `Vertex`, `Edge`, and `Face`.
/// Sequential integers assigned at construction time make all collection
/// iteration orders (BTreeMap, BTreeSet, sorted Vec) fully deterministic
/// across runs, eliminating the S3 non-determinism class of boolean bugs.
///
/// `SequentialID` is generic over a phantom type `T` for type safety:
/// a `SequentialID<Vertex>` cannot be compared with a `SequentialID<Edge>`.
pub struct SequentialID<T = ()>(u64, PhantomData<T>);

impl<T> Clone for SequentialID<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SequentialID<T> {}

impl<T> SequentialID<T> {
    /// Create a SequentialID from a raw counter value.
    #[inline]
    pub fn new(val: u64) -> Self {
        SequentialID(val, PhantomData)
    }

    /// Get the raw counter value.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl<T> PartialEq for SequentialID<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for SequentialID<T> {}

impl<T> Hash for SequentialID<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T> Ord for SequentialID<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> PartialOrd for SequentialID<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Debug for SequentialID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "seq#{}", self.0)
    }
}

impl<T> std::fmt::Display for SequentialID<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "seq#{}", self.0)
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

#[test]
fn seqid_monotonic() {
    reset_id_sequence();
    let id1: SequentialID = SequentialID::new(next_sequential_id());
    let id2: SequentialID = SequentialID::new(next_sequential_id());
    let id3: SequentialID = SequentialID::new(next_sequential_id());
    assert!(id1 < id2);
    assert!(id2 < id3);
    assert_eq!(id1.raw(), 1);
    assert_eq!(id2.raw(), 2);
    assert_eq!(id3.raw(), 3);
}

#[test]
fn seqid_reset_restarts_from_one() {
    reset_id_sequence();
    let id1: SequentialID = SequentialID::new(next_sequential_id());
    assert_eq!(id1.raw(), 1);
    reset_id_sequence();
    let id2: SequentialID = SequentialID::new(next_sequential_id());
    assert_eq!(id2.raw(), 1, "After reset, counter should restart from 1");
}

#[test]
fn seqid_debug_format() {
    let id: SequentialID = SequentialID::new(42);
    assert_eq!(format!("{id:?}"), "seq#42");
    assert_eq!(format!("{id}"), "seq#42");
}

#[test]
fn seqid_type_safety() {
    // Different phantom types produce incompatible IDs at compile time.
    // This test just verifies the basic mechanism works with explicit types.
    struct FakeVertex;
    struct FakeEdge;
    let v_id: SequentialID<FakeVertex> = SequentialID::new(1);
    let e_id: SequentialID<FakeEdge> = SequentialID::new(1);
    // These have the same raw value but different types.
    assert_eq!(v_id.raw(), e_id.raw());
    // v_id == e_id would be a compile error (different T).
}

#[test]
fn seqid_hash_and_eq() {
    use std::collections::HashSet;
    let mut set: HashSet<SequentialID> = HashSet::new();
    let id1: SequentialID = SequentialID::new(10);
    let id2: SequentialID = SequentialID::new(10);
    let id3: SequentialID = SequentialID::new(20);
    set.insert(id1);
    assert!(set.contains(&id2), "Equal IDs should hash and compare equal");
    assert!(!set.contains(&id3), "Different IDs should not collide");
}

#[test]
fn seqid_ord_matches_creation_order() {
    reset_id_sequence();
    let mut ids: Vec<SequentialID> = (0..50).map(|_| SequentialID::new(next_sequential_id())).collect();
    let original = ids.clone();
    ids.sort();
    assert_eq!(ids, original, "Sorted order should match creation order");
}
