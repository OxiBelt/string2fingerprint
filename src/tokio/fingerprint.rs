//! Fingerprints of input read from a Tokio [`AsyncRead`]: the asynchronous
//! counterpart of [`fingerprints`](crate::fingerprint::fingerprints) and
//! [`Fingerprinter`].
//!
//! A [`FingerprintReader`] reads its input in parts and returns the
//! fingerprint of each window as soon as the window is complete, without
//! holding the input. The windows equal those of
//! [`fingerprints`](crate::fingerprint::fingerprints) of the whole input,
//! however the reader splits it. With the low-level items of
//! [`crate::model`] they give what an
//! [`AsyncScorer`](super::AsyncScorer) gives:
//!
//! ```
//! use string2fingerprint::kernel::Kernel;
//! use string2fingerprint::model::{Logits, Model};
//! use string2fingerprint::tokio::fingerprint::FingerprintReader;
//! use string2fingerprint::Scorer;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> std::io::Result<()> {
//! let body: &[u8] = b"name=alice&bio=%3Cscript%3Ealert(1)%3C%2Fscript%3E";
//! let model = Model::embedded();
//! let kernel = Kernel::detect();
//! let mut reader = FingerprintReader::new(body);
//! let mut logits = Logits::NONE;
//! while let Some(window) = reader.next_window().await? {
//!     logits = logits.max_with(model.logits(kernel, window));
//! }
//! assert_eq!(logits.scores(), Scorer::new().score(body));
//! # Ok(())
//! # }
//! ```
//!
//! [`AsyncRead`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncRead.html

use core::fmt;
use core::future::poll_fn;
use core::pin::Pin;
use std::io;

use ::tokio::io::{AsyncRead, ReadBuf};

use crate::fingerprint::{BLOCK_LEN, Fingerprint, Fingerprinter};

/// The size of the buffer that a [`FingerprintReader`] reads into.
pub const READ_BUFFER_LEN: usize = 8192;

/// Reads an input from an [`AsyncRead`] and returns the fingerprint of each
/// of its windows in input order.
///
/// [`next_window`](Self::next_window) reads until a window is complete. At
/// the end of the input it returns the last window, as
/// [`Fingerprinter::finish`] does, and then `None`. The reader holds a
/// [`Fingerprinter`] (about 9 KiB) and a read buffer of
/// [`READ_BUFFER_LEN`] bytes on the heap, and allocates nothing else.
///
/// Fingerprinting a window takes a few microseconds, so the reader runs in
/// the polling task. Every window it returns takes one unit of the task's
/// [cooperative budget], so a reader that is always ready, such as a byte
/// slice, cannot keep a runtime worker to itself.
///
/// A reader that never ends is read forever. Bound an untrusted source,
/// for example with
/// [`AsyncReadExt::take`](https://docs.rs/tokio/1/tokio/io/trait.AsyncReadExt.html#method.take).
///
/// [`AsyncRead`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncRead.html
/// [cooperative budget]: https://docs.rs/tokio/1/tokio/task/coop/index.html
pub struct FingerprintReader<R> {
    reader: R,
    fingerprinter: Box<Fingerprinter>,
    buffer: Box<[u8]>,
    /// The unconsumed bytes of `buffer` are `start..end`.
    start: usize,
    end: usize,
    /// The number of bytes read since the reader was created.
    len: u64,
    /// The number of windows returned since the reader was created.
    windows: u64,
    /// Whether the input ended.
    done: bool,
    /// Whether the fingerprinter holds a window that was not returned yet.
    ready: bool,
}

/// Shows the counts and nothing derived from the input.
impl<R> fmt::Debug for FingerprintReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FingerprintReader")
            .field("len", &self.len)
            .field("windows", &self.windows)
            .field("done", &self.is_done())
            .finish_non_exhaustive()
    }
}

impl<R: AsyncRead + Unpin> FingerprintReader<R> {
    /// Creates a reader at the start of the input that `reader` holds.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            fingerprinter: Box::new(Fingerprinter::new()),
            buffer: vec![0; READ_BUFFER_LEN].into_boxed_slice(),
            start: 0,
            end: 0,
            len: 0,
            windows: 0,
            done: false,
            ready: false,
        }
    }

    /// Returns the fingerprint of the next window, or `None` once the
    /// input ended and every window was returned.
    ///
    /// Cancelling the call loses nothing: the bytes it read and a window it
    /// completed stay with the reader, and the next call goes on from them.
    ///
    /// # Errors
    ///
    /// Returns the first error of the inner reader other than
    /// [`io::ErrorKind::Interrupted`], which is retried. The reader keeps
    /// its state, so a later call can go on if the inner reader can.
    pub async fn next_window(&mut self) -> io::Result<Option<&Fingerprint>> {
        loop {
            if self.ready {
                // The window stays ready if the call is cancelled here.
                ::tokio::task::coop::consume_budget().await;
                self.ready = false;
                self.windows += 1;
                return Ok(Some(self.fingerprinter.last_window()));
            }
            if self.done {
                return Ok(None);
            }
            if self.start < self.end {
                // Up to the end of the current block, so that the update
                // completes at most one window.
                let filled = (self.fingerprinter.len() % BLOCK_LEN as u64) as usize;
                let take = (BLOCK_LEN - filled).min(self.end - self.start);
                let before = self.fingerprinter.windows();
                self.fingerprinter
                    .update(&self.buffer[self.start..self.start + take], |_| {});
                self.start += take;
                self.ready = self.fingerprinter.windows() != before;
                continue;
            }
            if self.fill().await? == 0 {
                self.done = true;
                self.ready = self.fingerprinter.finish().is_some();
            }
        }
    }

    /// Reads into the empty buffer and returns the number of bytes read,
    /// or 0 at the end of the input.
    async fn fill(&mut self) -> io::Result<usize> {
        loop {
            let mut buffer = ReadBuf::new(&mut self.buffer);
            let reader = &mut self.reader;
            match poll_fn(|cx| Pin::new(&mut *reader).poll_read(cx, &mut buffer)).await {
                Ok(()) => {
                    let read = buffer.filled().len();
                    self.start = 0;
                    self.end = read;
                    // A slice is shorter than `u64::MAX` bytes on every target.
                    self.len = self.len.saturating_add(read as u64);
                    return Ok(read);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
    }
}

impl<R> FingerprintReader<R> {
    /// Returns the number of bytes read from the inner reader.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Returns whether no byte was read from the inner reader.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of windows returned.
    #[must_use]
    pub const fn windows(&self) -> u64 {
        self.windows
    }

    /// Returns whether the input ended and every window was returned.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.done && !self.ready
    }

    /// Returns the inner reader.
    pub const fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Returns the inner reader. Reading from it directly skips those
    /// bytes in the fingerprints.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Returns the inner reader. Bytes that were read from it but not yet
    /// fingerprinted are dropped.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

#[cfg(test)]
mod tests {
    use core::task::{Context, Poll};

    use super::*;
    use crate::fingerprint::{MIN_INPUT_LEN, fingerprints};

    /// Returns its input in parts of `part` bytes, with an interrupted
    /// read before each part.
    struct Parts<'a> {
        input: &'a [u8],
        part: usize,
        interrupt: bool,
    }

    impl AsyncRead for Parts<'_> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
            buffer: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            self.interrupt = !self.interrupt;
            if self.interrupt && !self.input.is_empty() {
                return Poll::Ready(Err(io::ErrorKind::Interrupted.into()));
            }
            let take = self.part.min(self.input.len()).min(buffer.remaining());
            let (head, tail) = self.input.split_at(take);
            buffer.put_slice(head);
            self.input = tail;
            Poll::Ready(Ok(()))
        }
    }

    fn collect(input: &[u8], part: usize) -> Vec<Fingerprint> {
        ::tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime starts")
            .block_on(async {
                let mut reader = FingerprintReader::new(Parts {
                    input,
                    part,
                    interrupt: false,
                });
                let mut windows = Vec::new();
                while let Some(window) = reader.next_window().await.expect("no read fails") {
                    windows.push(window.clone());
                }
                assert!(reader.is_done());
                assert_eq!(reader.len(), input.len() as u64);
                assert_eq!(reader.windows(), windows.len() as u64);
                assert!(reader.next_window().await.expect("no read fails").is_none());
                windows
            })
    }

    #[test]
    fn windows_equal_those_of_the_whole_input() {
        let input: Vec<u8> = (0..3000u32)
            .map(|i| b"GET /a?q=<script>alert(1)</script>&x=../../etc/passwd "[i as usize % 52])
            .collect();
        for len in [
            0,
            1,
            MIN_INPUT_LEN,
            255,
            256,
            257,
            511,
            512,
            513,
            768,
            1024,
            3000,
        ] {
            let expected: Vec<Fingerprint> = fingerprints(&input[..len]).collect();
            for part in [1, 7, 256, 1000, READ_BUFFER_LEN] {
                assert_eq!(
                    collect(&input[..len], part),
                    expected,
                    "len {len}, part {part}"
                );
            }
        }
    }

    #[test]
    fn cancelled_calls_lose_no_window() {
        use core::future::Future;
        use core::pin::pin;

        // More windows than a task's budget of 128 units.
        let input: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let expected: Vec<Fingerprint> = fingerprints(&input).collect();
        let (windows, cancelled) = ::tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime starts")
            .block_on(async {
                let mut reader = FingerprintReader::new(&input[..]);
                let mut windows = Vec::new();
                let mut cancelled = 0;
                loop {
                    // Poll each call once and drop it if it is pending,
                    // which the spent budget makes it now and then.
                    let polled = {
                        let mut call = pin!(reader.next_window());
                        core::future::poll_fn(|cx| {
                            Poll::Ready(match call.as_mut().poll(cx) {
                                Poll::Ready(result) => {
                                    Some(result.expect("no read fails").cloned())
                                }
                                Poll::Pending => None,
                            })
                        })
                        .await
                    };
                    match polled {
                        Some(Some(window)) => windows.push(window),
                        Some(None) => break,
                        None => {
                            cancelled += 1;
                            ::tokio::task::yield_now().await;
                        }
                    }
                }
                (windows, cancelled)
            });
        assert!(cancelled > 0);
        assert_eq!(windows, expected);
    }

    #[test]
    fn a_read_error_is_returned_and_the_reader_goes_on() {
        struct Failing(bool);
        impl AsyncRead for Failing {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _: &mut Context<'_>,
                buffer: &mut ReadBuf<'_>,
            ) -> Poll<io::Result<()>> {
                if self.0 {
                    self.0 = false;
                    return Poll::Ready(Err(io::ErrorKind::ConnectionReset.into()));
                }
                let _ = buffer;
                Poll::Ready(Ok(()))
            }
        }
        ::tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime starts")
            .block_on(async {
                let mut reader = FingerprintReader::new(Failing(true));
                let error = reader
                    .next_window()
                    .await
                    .expect_err("the first read fails");
                assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
                assert!(
                    reader
                        .next_window()
                        .await
                        .expect("the input ends")
                        .is_none()
                );
                assert!(reader.is_empty());
            });
    }
}
