# Design

This document specifies what `string2fingerprint` computes and the contracts
its parts share. Rust documentation in `src/` is the reference for each public
item.

## Purpose and Scope

The library takes arbitrary bytes from an untrusted source, such as a request
line, header value, or body seen by a web reverse proxy, and returns one score
for each [category](#categories). A score says how much the input looks
malformed or looks like an attempt to escape the context it will be placed in.

The library scores text. It does not parse protocols, decode transport
encodings, block requests, or keep state between calls. The caller chooses
which fields to score and which thresholds to act on.

The library supports Linux on x86-64, AArch64, and RISC-V 64. Other operating
systems are rejected at compile time.

## Categories

| Index | Category | Describes |
| --- | --- | --- |
| 0 | `malformed` | Invalid UTF-8, NUL or other control bytes, invalid percent-escapes, binary data. |
| 1 | `obfuscation` | Repeated percent-encoding, `%u` escapes, overlong UTF-8, escape runs that spell ASCII, keyword splitting. |
| 2 | `sql_injection` | Escapes from an SQL string or numeric context. |
| 3 | `markup_injection` | Escapes into HTML or XML markup, including cross-site scripting. |
| 4 | `path_traversal` | Escapes from a directory, including encoded `../` sequences. |
| 5 | `command_injection` | Escapes into a shell command. |
| 6 | `template_injection` | Escapes into a template or expression language. |
| 7 | `header_injection` | Escapes from a header line with CR or LF. |

Categories are independent and an input can score high in several of them.
Indexes are stable. They index the scores and the model output layer.

## Input Contract

- The input is a byte slice. Any content and any length are accepted. Text is
  not required to be valid UTF-8, because invalid encodings are one of the
  things being scored.
- Scoring never fails and never panics on any input. The empty input and any
  input shorter than two bytes score zero in every category.
- Work is linear in the input length. Scoring allocates no heap memory and
  uses a fixed amount of stack. A caller that needs a hard time bound limits
  the length of the slice it passes.
- Scores depend only on the input bytes and the model. They do not depend on
  the CPU, the selected backend, or earlier calls.
- Diagnostics and errors never contain input bytes.

## Scores

A score is an integer from 0 to 255. Higher means the input looks more like
the category. Scores are model confidences, they are not calibrated
probabilities. Thresholds are the caller's choice and should be validated on
the caller's own traffic.

Inputs longer than one window are scored window by window. The score of the
input in a category is the highest score of any window in that category, so
padding an input with harmless text does not dilute a payload.

## Pipeline

```text
bytes -> blocks -> windows -> fingerprint -> int8 network -> scores
                                                              max over windows
```

### Fingerprint

A fingerprint is a fixed vector of 3072 unsigned bytes that summarizes one
window. It is built without interpreting the input as any particular language.

1. Each byte maps through two fixed tables to a *lexical* symbol and a *shape*
   symbol. Lexical symbols fold ASCII letter case and fold bytes at or above
   `0x80` into their UTF-8 roles. Shape symbols reduce a byte to one of 32
   classes: letter case and hex-digit membership, digit, whitespace and
   control groups, groups of punctuation by their part in escape syntax, and
   UTF-8 lead and continuation ranges that distinguish overlong and surrogate
   encodings.
2. N-grams of consecutive symbols are counted in five disjoint bucket ranges:

   | Buckets | N-gram | Bucket | Counted when |
   | --- | --- | --- | --- |
   | 0 to 383 | lexical bigram | hashed | at least one symbol is not alphanumeric |
   | 384 to 1023 | lexical trigram | hashed | at least one symbol is not alphanumeric |
   | 1024 to 1279 | lexical trigram | hashed | every symbol is alphanumeric |
   | 1280 to 2303 | shape bigram | one for each bigram | always |
   | 2304 to 3071 | shape trigram | hashed | always |

   Keeping punctuation-bearing n-grams apart from all-letter n-grams stops
   ordinary prose from crowding out the structural evidence of an escape.
   Shape n-grams expose structure that does not depend on the exact bytes,
   such as an invalid UTF-8 transition or a percent sign that is not followed
   by two hex digits. Every shape bigram has a bucket of its own, so that
   evidence of a malformed encoding never shares a bucket with an ordinary
   transition between letters, digits, and punctuation.
3. The input is cut into blocks of 256 bytes. An n-gram belongs to the block
   that holds its last byte, so n-grams that straddle a block edge are kept.
   Each block has a histogram of bucket counts.
4. A window is two adjacent blocks, and consecutive windows share one block.
   An input of at most one block is one window. Any run of up to 257 bytes
   therefore lies completely inside at least one window.
5. Window counts are capped at 15 and mapped through a fixed logarithmic table
   to the range from 0 to 255. The cap bounds how far repeating one n-gram can
   move a score.

Fingerprint extraction is portable integer code and gives the same result on
every target.

### Network

The fingerprint feeds a small fully connected network with quantized
parameters:

- Activations are unsigned 8-bit, weights are signed 8-bit in the range from
  -127 to 127, and biases and accumulators are signed 32-bit.
- A layer computes `acc[r] = bias[r] + sum(weight[r][c] * input[c])`, then
  `(acc[r] * multiplier[r] + 2^(shift - 1)) >> shift` in 64-bit arithmetic.
  Hidden layers clamp the result to the range from 0 to 255, which is also the
  rectifier. The output layer clamps to the range from -128 to 127 and maps
  the result through a fixed logistic table to a score.
- A model is rejected when loaded unless every accumulator provably fits in 32
  bits for every possible input, so inference has no overflow case.

The dot product of unsigned 8-bit activations and signed 8-bit weights is the
only operation with CPU-specific implementations.

### Model Format

A model is a little-endian byte string: a magic number and version, the layer
dimensions, the parameters of each layer, and a checksum. The input dimension
must equal the fingerprint length and the output dimension must equal the
number of categories. Hidden dimensions and the layer count are bounded.
Parsing checks the length and the checksum first, so it allocates at most the
size of the byte string. It validates every field and reports a fixed error
kind without echoing model bytes. The checksum detects accidental corruption.
It is not an authenticity check, so load models only from a trusted source.

## Backends

A backend is one implementation of the dot-product kernel. All backends give
bit-identical results. One is chosen when a scorer is created, in this order
of preference, using the first whose CPU features are all present:

| Architecture | Order |
| --- | --- |
| x86-64 | AVX-512 VNNI, AVX-VNNI, AVX2, scalar |
| AArch64 | SVE with I8MM, NEON with I8MM, NEON with DotProd, NEON, scalar |
| RISC-V 64 | Zvdot4a8i (reserved), RVV 1.0, scalar |

Rules that apply to every backend:

- CPU features are detected at run time. Default builds enable no CPU feature
  globally, and target-specific instructions execute only after every feature
  they need was detected.
- A caller can request a specific backend. The request fails when the CPU
  does not support it.
- A backend that the operating system cannot report yet is never selected
  automatically. It is available only through an explicit `unsafe` opt-in in
  which the caller vouches for the CPU.
- Zvdot4a8i is reserved. The extension is not ratified, Linux has no
  interface that reports it, and a kernel for it ships only once it can be
  validated on an emulator or on hardware. Until then the backend is never
  supported, and requesting it without detection runs the scalar kernel.
- On RISC-V, vector use is detected from the `V` bit of `AT_HWCAP`, which
  Linux clears when vector use is disabled for the process.
- The scalar backend is safe Rust and is the reference for every other
  backend. Tests compare each available backend with it.

## Security Considerations

- Inputs are attacker-controlled. Every loop is bounded by the input length,
  every table index is derived from a byte or a reduced hash, and no input
  byte selects an amount of memory.
- Bucket hashing uses fixed constants. An attacker who knows the model can
  search for inputs that score low, as with any published detector. The count
  cap and per-window maximum limit dilution, they do not prevent evasion.
- Treat scores as one signal among several, and measure false positives on
  your own traffic before enforcing on them.
- Unsafe Rust is confined to the target-specific kernel modules and to CPU
  feature queries. Each unsafe operation states the invariant it relies on.
