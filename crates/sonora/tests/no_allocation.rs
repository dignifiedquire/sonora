//! Echo cancellation runs in real-time audio callbacks, where allocating
//! memory can block. This test checks that steady-state processing with the
//! echo canceller enabled does not allocate. It counts allocations of the
//! current thread only, so tests running in parallel do not interfere.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    if COUNTING.with(Cell::get) {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
    }
}

struct CountingAllocator;

// SAFETY: forwards every call unchanged to the system allocator and only
// counts calls.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Deterministic noise in [-1, 1] (xorshift64).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

#[test]
fn echo_cancellation_does_not_allocate_in_steady_state() {
    const RATE: u32 = 48_000;
    const FRAME: usize = 480;
    let stream = StreamConfig::new(RATE, 1);
    let mut apm = AudioProcessing::builder()
        .config(Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Config::default()
        })
        .capture_config(stream)
        .render_config(stream)
        .build();

    // Far end with pauses, an echo delayed by 60 ms and near-end talk every
    // fourth second (double talk), so that all echo canceller paths run.
    let frames = 1500;
    let mut noise = Noise(0x1234_5678_9abc_def1);
    let far: Vec<f32> = (0..frames * FRAME)
        .map(|n| {
            let active = ((n as f32 / RATE as f32) * 9.4).sin() > -0.3;
            if active { 0.3 * noise.next() } else { 0.0 }
        })
        .collect();
    let mic: Vec<f32> = (0..far.len())
        .map(|n| {
            let echo = if n >= 2880 { 0.3 * far[n - 2880] } else { 0.0 };
            let near = if (n / RATE as usize) % 4 == 3 {
                0.2 * noise.next()
            } else {
                0.0
            };
            echo + near + 0.001 * noise.next()
        })
        .collect();

    let mut render_out = [0.0f32; FRAME];
    let mut capture_out = [0.0f32; FRAME];
    for f in 0..frames {
        // Allow lazy initialisation during the first 5 seconds.
        if f == 500 {
            COUNTING.with(|c| c.set(true));
        }
        let range = f * FRAME..(f + 1) * FRAME;
        apm.process_render_f32(&[&far[range.clone()]], &mut [&mut render_out])
            .unwrap();
        apm.process_capture_f32(&[&mic[range]], &mut [&mut capture_out])
            .unwrap();
        let _ = apm.statistics();
    }
    COUNTING.with(|c| c.set(false));

    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
