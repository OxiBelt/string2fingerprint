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
| 0 | `malformed` | Invalid UTF-8, including text in a legacy or UTF-16 encoding, NUL or other control bytes, invalid percent-escapes, binary data. |
| 1 | `obfuscation` | Repeated percent-encoding, `%u` escapes, overlong UTF-8, escape runs or character references that spell ASCII, UTF-7, look-alike, invisible, or decorated characters, keyword splitting. |
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

The scores of each window, with the input bytes it covers, show where in a
long input the evidence lies. An input that arrives in parts scores the same
as the whole input, however it is split. While it arrives, the highest score
of the windows completed so far never decreases and never exceeds the final
score, so a caller can act before the input ends. The fields of one message
can be scored apart and combined by their highest score in each category, so
that no window spans two fields.

### Thresholds

A threshold turns a score into a decision. A threshold of `t` flags every
score that is at least `t`, and the threshold `never` flags no score, which
turns a category off. A caller can hold one threshold for each category,
because categories differ in how well a model separates them.

A detector holds a scorer and a threshold for each category, and returns the
scores of an input together with the categories they flag. A preset pairs an
embedded model with fixed thresholds, and the rates at which each preset flags
attack and benign samples were measured on the synthetic test split of the
training tools. Presets are starting points: thresholds should still be
validated on the caller's own traffic.

A set of categories is one bit for each category, by index. It names the
categories that are flagged, or the categories that label a sample, and the
empty set labels benign text. For one input, the flagged categories that are
not labels are its false positives, and the labels that are not flagged are
its false negatives.

## Evaluation

An evaluation measures a model on labeled samples. A sample is the scores of
one input and the set of its labels. A sample is a positive of each category
in its labels and a negative of every other category. At a threshold, a
positive that is flagged is a true positive and one that is not is a false
negative, and a negative that is flagged is a false positive and one that is
not is a true negative.

The benign samples are those whose label set is empty. When only benign
samples count as negatives, a sample labeled with other categories is no
longer a negative of a category, and it still counts as a positive of its own
categories. This view applies when a flag in the wrong attack category is not
an error that matters.

The sample-level view treats each sample as one decision. A sample is a
positive when its label set is not empty, and it is flagged at one threshold
for every category when its highest score reaches that threshold.

An evaluation keeps the number of positive and negative samples at each
score for each category, so that every threshold can be evaluated after the
samples are recorded. Counts saturate instead of overflowing. The rate of `k`
of `n` samples is the double-precision quotient of the two counts, which is
the nearest double-precision value to `k / n` when `n` is below 2^53. A limit
on a rate is met when that value is within the limit, so a limit that equals
`k / n` in decimal is met by `k` of `n`.

A threshold can be chosen for each category to meet one objective:

- A maximum false positive rate chooses the lowest threshold whose false
  positive rate is at most the limit. The threshold `never` has the rate 0. It
  is undefined without a negative sample or for a limit outside 0 to 1.
- A minimum recall chooses the highest threshold whose recall is at least the
  limit. The threshold 0 has the recall 1. It is undefined without a positive
  sample or for a limit outside 0 to 1.
- A minimum cost chooses the threshold with the least sum of the false
  positives and the false negatives, each weighted by its cost. When several
  thresholds have the least cost, it chooses the lower median of them in
  ascending order, so that a gap between the scores of the positives and the
  negatives is split in the middle. It is undefined without a sample or for a
  cost that is negative or not finite.

A category whose objective is undefined keeps a threshold that the caller
supplies.

Results describe only the samples that were recorded. Thresholds should be
chosen on samples that were not used to train the model, and checked on
samples that were not used to choose them.

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
   An input that arrives in parts gives the same windows however it is split.
   A window is complete when its second block is, and the last window when
   the input ends.
5. Window counts are capped at 15 and mapped through a fixed logarithmic table
   to the range from 0 to 255. The cap bounds how far repeating one n-gram can
   move a score.

Fingerprint extraction is portable integer code and gives the same result on
every target.

The parts of the extraction are public: the symbol of a byte in each table,
the section of features that counts each kind of n-gram, the n-grams of an
input in order with the feature that counts each one, and the input bytes of
each window. The feature of a window is then the compressed number of
n-grams of its blocks with that feature, so a caller can trace a feature back
to the bytes that raised it. These parts are the layout, and a change to any
of them needs every model to be retrained.

The layout has a [version](#versions). Its major number changes when any input
can get a different fingerprint. Its minor number changes when features are
added that were zero in every fingerprint of the earlier layout, so that
earlier models ignore them. Its patch number changes when no fingerprint
changes. A model of layout `m` is compatible with the layout of the crate when
that layout matches the requirement `^m`: it is not lower than `m` and has the
same major number. The layout version stays at 1.0.0 or above and has no
pre-release, because below 1.0.0 the caret requirement would refuse older
models at every minor change, and a pre-release would match only models of
its own release. The current layout is 1.0.0, which is also the layout of
every earlier revision of the crate.

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

The clamped values of the output layer are the logits of the model, in units
of 1/16, and the logistic table turns them into scores. A caller can read the
logits of an input, which are the highest of its windows in each category,
and the accumulators and activations of every layer for one fingerprint. The
same code computes them and the scores, so they agree on every backend.

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
A model has exactly one encoding, so writing a parsed model gives the bytes
it was parsed from. A model can also be created from the parameters of its
layers, which are validated in the same way.

The crate embeds several models in this format, which differ in their
hidden layers and in the weight that training gave to benign text. One of
them is the default. Their training procedure, data, and measured limits are
documented with the [training tools](../tools/train/README.md).

## Training

The `train` feature adds a trainer to the library, so that a program that
depends on the library can produce a model from its own labeled samples. The
feature adds no dependency. Training data is trusted input: a dataset and a
trainer allocate memory in proportion to the samples they are given.

### Samples

A sample is the bytes of one input and the set of its labels. A dataset
stores the fingerprint of every window of a sample, extracted by the same
code that scoring uses, and not the bytes. An input without a window is
skipped.

A model scores one window at a time, and training teaches it the labels of a
sample for each of its windows. A payload usually lies in one window of a
long input, so its other windows would teach the model to flag the harmless
text around it. A labeled input with more than one window is therefore
refused unless the caller states that every window shows the labels. A benign
input of any length is accepted, and each of its windows is benign.

A sample has a weight, 1 unless the caller sets another, which multiplies the
loss of each of its windows. A weight must be finite and not negative.

The score of a sample is the highest score of any of its windows in each
category, which is the score of its input.

### Procedure

The trainer optimizes a float network with the shape of the quantized one.
Hidden activations are clamped to the range from 0 to 1, which is the range
of a stored activation. The loss of a window is the sum over the categories
of the binary cross-entropy of the logit with the label, so that categories
stay independent. The optimizer is Adam with decoupled weight decay and a
step size that falls to zero along a cosine. A share of the features of each
window can be set to zero at random.

In its last epochs the forward pass rounds the weights of each output to the
255 levels that they will have, and rounds activations to multiples of 1/255.
Gradients pass straight through the rounding. The network adapts to the
quantization that follows.

Quantization gives each output of a layer its own weight scale: the largest
weight magnitude of the output maps to 127. The bias is stored in units of
1/255 of that scale, which is the unit of the accumulator. The multiplier and
the shift of the layer then convert the accumulator to a stored activation,
or to a logit in units of 1/16 in the output layer. The shift is the largest
for which no multiplier of the layer is above 2^30. The result is validated
like any other model before it is returned.

### Costs

The loss has one term for each category of each window. The term of a
category that labels the window raises the score, and its weight is the cost
of a false negative in that category. The term of a category that does not
label the window lowers the score, and its weight is the cost of a false
positive in that category. The terms of a window without any label are also
multiplied by the benign weight. Each term is then multiplied by the weight
of its sample. All costs are 1 by default, and a cost must be finite and not
negative.

A higher cost makes the model avoid that error at the expense of the opposite
one. Costs change what the model learns, and a threshold chosen on held-out
samples then selects the operating point. Costs do not replace that
threshold.

### Fine-Tuning

Fine-tuning continues the training of a model on new samples. The float
network starts from the stored parameters: a weight is its stored value
multiplied by the scale of its output, and the scale follows from the
multiplier and the shift. A base whose parameters the trainer could not store
again is refused. The layers stay those of the base model. Weight decay pulls
each weight towards its value in the base model, not towards zero.

A tuned model changes its scores on every text that shares features with the
new samples. Benign samples alone lower the scores of attacks that look like
them. The new samples must therefore include text whose scores must stay,
and the result must be measured on held-out samples of both kinds. The
[training tools](../tools/train/README.md#correcting-a-model) record a
measurement of both effects.

### Training While Compiling

The `build` feature, which includes `train`, lets the build script of a
dependent crate train its model from data files of that crate. The model is
then trained again whenever a data file changes, and never at run time. The
build script depends on the library with the feature, and the crate itself
depends on it without the feature. With feature resolver 2 or later, Cargo
builds the two apart, so the compiled crate contains no training code.

A build reads the files, trains or fine-tunes, and writes the model bytes to
`<name>.s2fp` in the output directory that Cargo gives the build script. It
can hold back a share of the samples, evaluate the model on them and on
validation files, and write thresholds chosen on them to
`<name>.thresholds` in the form of `Thresholds::to_bytes`. Two macros that
need no feature embed both files in the dependent crate and validate them
when they are loaded. Files are written only after everything else
succeeded, and a build without thresholds removes those of an earlier build,
which Cargo would otherwise keep.

The build tells Cargo to run it again when a data file or one of its
environment variables changes, so every path must be UTF-8 without a line
break. A name is lowercase ASCII letters, digits, and `_`, which gives each
name a variable of its own. `STRING2FINGERPRINT_MODEL_<NAME>` names a model
file that is validated and used in place of training, with a warning, so
that a crate can be built with a model trained elsewhere or without its
data.
`STRING2FINGERPRINT_TRAIN_THREADS` sets the number of threads, which
otherwise follows the number of jobs of the build. Training is
deterministic, so neither changes the model.

The build host must be Linux, because the build script links the library.
Data files and the variables are trusted input: whoever controls them
chooses what the model flags.

### Determinism

The same samples in the same order with the same settings give the same
model bytes, on every supported host and with any number of threads. The
trainer uses only operations whose result IEEE 754 or Rust defines exactly:
additions, multiplications, divisions, square roots, remainders,
comparisons, rounding to an integer, and conversions between integers and
floats. Its exponential, logarithm, and cosine are its own, built from
them. A batch is divided into a fixed number of parts whose
gradients are added in a fixed order, whatever thread computed them. A test
holds the checksum of a model that a fixed run must produce.

## Versions

A version has the form of
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): three
numbers without leading zeros, an optional pre-release, and optional build
metadata, at most 64 bytes in all. Versions are ordered by the precedence of
that specification, and build metadata, which precedence ignores, only breaks
ties.

A version requirement has the matching rules of Cargo and a subset of its
syntax, without the spellings that Cargo rewrites into others: `x` for `*`, an
operator before a wildcard, `1.*.*`, and build metadata. It is
`*` or a list of comparators separated by commas. A comparator is an operator
(`=`, `>`, `>=`, `<`, `<=`, `~`, or `^`) followed by a version whose minor and
patch numbers may be left out, a version without an operator, which means
`^`, or a wildcard in place of the minor or patch number, such as `1.*`. Build
metadata is not part of a requirement. A version matches when it matches every
comparator. A version with a pre-release also needs a comparator with the same
major, minor, and patch numbers and a pre-release, so that a pre-release
matches only a requirement that names its release. The canonical text of a
requirement writes operators without spaces, writes `^` out, and separates
comparators with a comma and a space. A requirement has at most 32
comparators, the limit of Cargo, and its canonical text is at most 255 bytes
long. Only spaces may separate the tokens of a requirement.

## Backends

A backend is one implementation of the dot-product kernel. All backends give
bit-identical results. One is chosen when a scorer is created, in this order
of preference, using the first whose CPU features are all present:

| Architecture | Order |
| --- | --- |
| x86-64 | AVX-512 VNNI, AVX-VNNI, AVX2, scalar |
| AArch64 | SVE with I8MM, NEON with I8MM, NEON with DotProd, NEON, scalar |
| RISC-V 64 | Zvdot4a8i (experimental, never detected), RVV 1.0, scalar |

Rules that apply to every backend:

- CPU features are detected at run time. Default builds enable no CPU feature
  globally, and target-specific instructions execute only after every feature
  they need was detected.
- A caller can request a specific backend. The request fails when the CPU
  does not support it.
- A caller can run the kernel of a backend on a matrix of its own, under the
  same rules. The matrix is validated when it is created, the lengths of the
  vectors are checked on each call, and a sum wraps at 32 bits on every
  backend.
- A backend that the operating system cannot report yet is never selected
  automatically. It is available only through an explicit `unsafe` opt-in in
  which the caller vouches for the CPU.
- Zvdot4a8i is experimental. The extension is not ratified and Linux has no
  interface that reports it, so the backend is never detected and runs only
  through the `unsafe` opt-in. Its kernel uses `vdot4asu.vv` with the
  encoding and semantics of the draft as the Spike simulator implemented it
  under the name `zvdot4a` at revision `6e18887`. The documentation of the
  backend states them. The kernel was executed on that simulator only, at
  vector lengths from 128 to 4096 bits, and on no hardware. A ratified
  extension can differ from the draft.
- A scorer can check the kernel of its backend against the scalar kernel on
  built-in matrices. A caller who opts in to a backend without detection
  should run that self-test first. The `kernel_self_test` example runs it
  from the command line. The self-test does not drive a single 32-bit lane
  of a vector kernel beyond its range, which no valid model does either.
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
- The default model is trained on synthetic data and has not been measured on
  real traffic. Treat scores as one signal among several, and measure false
  positives on your own traffic before enforcing on them.
- Unsafe Rust in the library is confined to the target-specific kernel
  modules, to CPU feature queries, and to the opt-in that selects a backend
  without detection. The `kernel_self_test` example calls that opt-in. Each
  unsafe operation states the invariant it relies on.
