//! Runs CPU-bound work in the polling task or on Tokio's blocking pool.

use core::fmt;
use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, ready};
use std::collections::VecDeque;
use std::panic;
use std::sync::{Arc, Mutex, PoisonError};

use ::tokio::runtime::Handle;
use ::tokio::task::{JoinHandle, coop};

/// Work that a [`Job`] runs once.
type Work<O> = Box<dyn FnOnce() -> O + Send>;

/// Holds the work of a job on the blocking pool until a thread of the pool
/// takes it. If the runtime shuts down first, the pool drops its closure
/// without running it and the work is still here.
type Slot<O> = Arc<Mutex<Option<Work<O>>>>;

/// Takes the work out of `slot`. No code panics while holding the lock,
/// so a poisoned lock still holds a consistent value.
fn take<O>(slot: &Slot<O>) -> Option<Work<O>> {
    slot.lock().unwrap_or_else(PoisonError::into_inner).take()
}

/// A future that runs some work and returns its result.
///
/// Work on the blocking pool starts when the job is created and goes on if
/// the job is dropped after a thread of the pool took it; its result is then
/// discarded. Dropping the job before that drops the work unrun. Work in the
/// polling task runs at the first poll that the task's cooperative budget
/// allows.
pub(crate) struct Job<O> {
    state: State<O>,
}

enum State<O> {
    /// Runs in the polling task. The mutex makes the job `Sync`; it is only
    /// reached through `&mut`, with [`Mutex::get_mut`], so it never locks.
    Inline(Mutex<Option<Work<O>>>),
    /// Runs on the blocking pool of a runtime.
    Blocking {
        handle: JoinHandle<Option<O>>,
        slot: Slot<O>,
    },
}

impl<O: Send + 'static> Job<O> {
    /// Runs `work` in the polling task when `cost` is at most
    /// `inline_limit` and on the blocking pool otherwise.
    pub(crate) fn sized(
        cost: usize,
        inline_limit: usize,
        work: impl FnOnce() -> O + Send + 'static,
    ) -> Self {
        if cost <= inline_limit {
            Self::inline(work)
        } else {
            Self::blocking(work)
        }
    }

    /// Runs `work` in the polling task.
    pub(crate) fn inline(work: impl FnOnce() -> O + Send + 'static) -> Self {
        Self {
            state: State::Inline(Mutex::new(Some(Box::new(work)))),
        }
    }

    /// Starts `work` on the blocking pool of the current runtime, or runs
    /// it in the polling task when the caller is outside a runtime.
    pub(crate) fn blocking(work: impl FnOnce() -> O + Send + 'static) -> Self {
        let work: Work<O> = Box::new(work);
        let Ok(runtime) = Handle::try_current() else {
            return Self {
                state: State::Inline(Mutex::new(Some(work))),
            };
        };
        let slot = Arc::new(Mutex::new(Some(work)));
        let shared = Arc::clone(&slot);
        let handle = runtime.spawn_blocking(move || take(&shared).map(|work| work()));
        Self {
            state: State::Blocking { handle, slot },
        }
    }
}

// No field is pinned: the job only moves its output out.
impl<O> Unpin for Job<O> {}

impl<O> Drop for Job<O> {
    /// Drops work that no thread of the pool took yet, so that work for a
    /// caller that went away does not hold the pool.
    fn drop(&mut self) {
        if let State::Blocking { slot, .. } = &self.state {
            drop(take(slot));
        }
    }
}

impl<O: Send + 'static> Future for Job<O> {
    type Output = O;

    /// # Panics
    ///
    /// When the work panicked, with its payload, and when polled after it
    /// returned its result.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<O> {
        match &mut self.get_mut().state {
            State::Inline(work) => {
                let coop = ready!(coop::poll_proceed(cx));
                let work = work
                    .get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .expect("`Job` polled after completion");
                coop.made_progress();
                Poll::Ready(work())
            }
            State::Blocking { handle, slot } => {
                let result = ready!(Pin::new(handle).poll(cx));
                Poll::Ready(match result {
                    Ok(Some(output)) => output,
                    Err(error) if error.is_panic() => panic::resume_unwind(error.into_panic()),
                    // The runtime shut down before a thread of the pool took
                    // the work, so it is still in the slot: run it here.
                    _ => take(slot).expect("`Job` polled after completion")(),
                })
            }
        }
    }
}

/// The future of some scoring, deciding, or loading, which runs in the
/// polling task or on Tokio's blocking pool. See the [module
/// documentation](super#where-work-runs).
///
/// It owns its input and borrows nothing, and is `Send`, `Sync`, and
/// `Unpin` when its output is `Send`. Dropping it before a thread of the
/// pool took its work drops the work unrun; work that started finishes,
/// and its result is dropped.
#[must_use = "futures do nothing unless polled; work on the blocking pool is dropped with them"]
pub struct Offload<T> {
    job: Job<T>,
}

impl<T> Offload<T> {
    pub(crate) const fn new(job: Job<T>) -> Self {
        Self { job }
    }
}

/// Shows nothing derived from the input.
impl<T> fmt::Debug for Offload<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Offload").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Future for Offload<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        Pin::new(&mut self.job).poll(cx)
    }
}

/// The future of a batch scored or decided in several jobs on Tokio's
/// blocking pool at once, which returns the results in input order.
///
/// It is `Send` and `Unpin` when its items are `Send`, `Sync` when they
/// are also `Sync`, and drops work as [`Offload`] does.
#[must_use = "futures do nothing unless polled; work on the blocking pool is dropped with them"]
pub struct Parallel<T> {
    /// The jobs whose results are not in `results` yet, in input order.
    jobs: VecDeque<Job<Vec<T>>>,
    results: Vec<T>,
}

impl<T> Parallel<T> {
    pub(crate) fn new(jobs: Vec<Job<Vec<T>>>) -> Self {
        Self {
            jobs: jobs.into(),
            results: Vec::new(),
        }
    }
}

// No field is pinned: the future only moves the results out.
impl<T> Unpin for Parallel<T> {}

/// Shows the progress and nothing derived from the input.
impl<T> fmt::Debug for Parallel<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Parallel")
            .field("pending_jobs", &self.jobs.len())
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Future for Parallel<T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Vec<T>> {
        let this = self.get_mut();
        while let Some(job) = this.jobs.front_mut() {
            let share = ready!(Pin::new(job).poll(cx));
            this.jobs.pop_front();
            if this.results.is_empty() {
                this.results = share;
            } else {
                this.results.extend(share);
            }
        }
        Poll::Ready(mem::take(&mut this.results))
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    fn runtime() -> ::tokio::runtime::Runtime {
        ::tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime starts")
    }

    #[test]
    fn inline_work_runs_in_the_polling_thread() {
        let caller = thread::current().id();
        let job = Job::sized(4, 4, move || thread::current().id() == caller);
        assert!(runtime().block_on(job));
    }

    #[test]
    fn large_work_runs_on_the_blocking_pool() {
        let caller = thread::current().id();
        let job = runtime().block_on(async move {
            Job::sized(5, 4, move || thread::current().id() != caller).await
        });
        assert!(job);
    }

    #[test]
    fn blocking_work_outside_a_runtime_runs_when_polled() {
        let caller = thread::current().id();
        // Created outside any runtime, then polled inside one.
        let job = Job::blocking(move || thread::current().id() == caller);
        assert!(runtime().block_on(job));
    }

    #[test]
    fn work_survives_a_runtime_that_shut_down() {
        let caller = thread::current().id();
        let shut_down = runtime();
        let handle = shut_down.handle().clone();
        drop(shut_down);
        // The pool of a runtime that shut down cancels the job at once.
        let job = {
            let _context = handle.enter();
            Job::blocking(move || thread::current().id() == caller)
        };
        assert!(matches!(job.state, State::Blocking { .. }));
        // Awaiting it runs the work here.
        assert!(runtime().block_on(job));
    }

    #[test]
    fn dropping_a_queued_job_drops_its_work() {
        let runtime = ::tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .build()
            .expect("a runtime starts");
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        runtime.block_on(async move {
            // Occupy the only thread of the pool, so the next job queues.
            let busy = Job::blocking(move || {
                let _ = wait.recv();
            });
            drop(Job::blocking(move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst)
            }));
            drop(release);
            busy.await;
            // The queued closure ran after the busy one and found no work.
            Job::blocking(|| ()).await;
        });
        assert!(!ran.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn futures_are_send_sync_and_unpin() {
        fn check<T: Send + Sync + Unpin + 'static>() {}
        check::<Offload<u8>>();
        check::<Parallel<u8>>();
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn a_panic_of_the_work_reaches_the_caller() {
        runtime().block_on(async { Job::blocking(|| -> u8 { panic!("boom") }).await });
    }
}
