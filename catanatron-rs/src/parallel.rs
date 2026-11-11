//! Running many independent games across threads.
//!
//! Work is handed out in small chunks of indices from a shared counter, so
//! threads stay busy even though game lengths vary. Results come back in
//! index order, so output never depends on the thread count or scheduling,
//! as long as `f(i)` depends only on `i`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Most indices claimed per grab from the shared counter.
const MAX_CHUNK: u64 = 8;

/// Indices per grab: small enough that every thread gets work and the last
/// grabs even out (about 8 per thread), at most MAX_CHUNK (contention on the
/// counter only matters for many short tasks). With chunks of 8, a batch of
/// 4 long games per thread kept only half the threads busy.
fn chunk_size(n: u64, threads: usize) -> u64 {
    (n / (threads as u64 * 8)).clamp(1, MAX_CHUNK)
}

/// Default thread count: all available hardware threads.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Computes `f(0), f(1), ..., f(n - 1)` on `threads` threads and returns the
/// results in index order.
pub fn par_map<T, F>(n: u64, threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(u64) -> T + Sync,
{
    let threads = threads.max(1).min(n.max(1) as usize);
    if threads == 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = chunk_size(n, threads);
    let next = AtomicU64::new(0);
    let mut parts: Vec<Vec<(u64, T)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let start = next.fetch_add(chunk, Ordering::Relaxed);
                        if start >= n {
                            break;
                        }
                        for i in start..(start + chunk).min(n) {
                            out.push((i, f(i)));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    let mut all: Vec<(u64, T)> = parts.iter_mut().flat_map(std::mem::take).collect();
    all.sort_unstable_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, t)| t).collect()
}
