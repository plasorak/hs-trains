#![allow(dead_code)] // placeholder fields will be read once events drive real logic

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// ---------------------------------------------------------------------------
// Entity references
// ---------------------------------------------------------------------------

/// A reference to a simulated entity that can own or be the target of an event.
/// Add new variants here as infrastructure types are introduced.
#[derive(Debug, Clone, Copy)]
pub enum EntityRef {
    Train(usize),
    Signal(usize),
}

// ---------------------------------------------------------------------------
// Cancellation token
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`EventQueue::push`], usable to cancel the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(u64);

// ---------------------------------------------------------------------------
// Event kinds
// ---------------------------------------------------------------------------

/// Placeholder random event kinds — no simulation effect yet.
/// The entity they act on is carried by `Event::target`, not embedded here.
#[derive(Debug, Clone)]
pub enum RandomEventKind {
    Departure,
    Arrival,
    SignalChange,
    SpeedChange { new_speed_kmh: f64 },
}

#[derive(Debug, Clone)]
pub enum EventKind {
    /// Advance physics by one fixed time step.
    /// Self-scheduling: each tick pushes the next one; the event loop's time
    /// guard (`event.time > duration`) stops the chain.
    PhysicsTick,
    /// Placeholder event targeting an entity — no effect yet.
    Random(RandomEventKind),
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Event {
    pub id: EventId,
    /// Simulation time (seconds) at which this event fires.
    pub time: f64,
    /// The entity this event acts on, if any.
    pub target: Option<EntityRef>,
    pub kind: EventKind,
}

// Min-heap ordering: smaller time → higher priority (via BinaryHeap's max-heap).
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// EventQueue
// ---------------------------------------------------------------------------

/// Priority queue of simulation events ordered by ascending time.
///
/// Cancellation is lazy: cancelled events are silently skipped on [`pop`]
/// rather than being removed from the heap immediately (O(1) cancel, O(log n) pop).
pub struct EventQueue {
    heap: BinaryHeap<Event>,
    cancelled: HashSet<EventId>,
    next_id: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            cancelled: HashSet::new(),
            next_id: 0,
        }
    }

    /// Schedule a new event. Returns an [`EventId`] that can be passed to
    /// [`cancel`] to prevent the event from firing.
    pub fn push(&mut self, time: f64, target: Option<EntityRef>, kind: EventKind) -> EventId {
        let id = EventId(self.next_id);
        self.next_id += 1;
        self.heap.push(Event {
            id,
            time,
            target,
            kind,
        });
        id
    }

    /// Mark an event as cancelled. It will be silently discarded when it
    /// reaches the front of the queue. Safe to call on an already-fired or
    /// already-cancelled event (no-op in both cases).
    pub fn cancel(&mut self, id: EventId) {
        self.cancelled.insert(id);
    }

    /// Pop and return the earliest non-cancelled event, or `None` if the
    /// queue is empty (or contains only cancelled events).
    pub fn pop(&mut self) -> Option<Event> {
        loop {
            let event = self.heap.pop()?;
            // `remove` returns true if the id was present → event is cancelled.
            if !self.cancelled.remove(&event.id) {
                return Some(event);
            }
            // Cancelled — discard and try the next one.
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}
