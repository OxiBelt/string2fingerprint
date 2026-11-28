#!/usr/bin/env bash
# Builds the image of `tests/docker/Dockerfile` and checks, end to end, the
# programs in it: a dependent crate whose build script trained its model,
# scoring with every embedded model, deciding with every preset through the
# synchronous and the Tokio interface, training and fine-tuning from the
# command line, and failures on bad input.
#
# Usage: tests/docker/e2e.sh [--digests FILE]
#
# Every container runs without a network, as a user without privileges, on
# a read-only file system with a small writable `/tmp`. `--digests FILE`
# writes the SHA-256 of every output that must not depend on the CPU, one
# `NAME DIGEST` line each, so that the outputs of two hosts can be compared.
# `S2FP_E2E_IMAGE` names the image, `string2fingerprint-e2e:local` by
# default. Needs Docker, Python 3, and coreutils.

set -euo pipefail

usage() {
  echo "usage: tests/docker/e2e.sh [--digests FILE]" >&2
  exit 2
}

digests=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --digests)
      [[ $# -ge 2 ]] || usage
      digests="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${S2FP_E2E_IMAGE:-string2fingerprint-e2e:local}"
data=/usr/local/share/string2fingerprint
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

pass() {
  echo "ok: $*"
}

# Runs a command in a container of the image, with standard input passed
# through.
run() {
  docker run --rm -i \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m,mode=1777 \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user 65532:65532 \
    "$image" "$@"
}

# Runs a command in a container, with standard input from `$stdin` or
# empty, and checks its exit status and that it did not panic.
expect_status() {
  local expected="$1" what="$2"
  shift 2
  local status=0
  run "$@" < "${stdin:-/dev/null}" > "$work/failure.out" 2> "$work/failure.err" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    fail "$what exited with status $status, not $expected"
  elif grep -q 'panicked' "$work/failure.err"; then
    fail "$what panicked"
  else
    pass "$what exits with status $expected"
  fi
}

# Records the digest of a file that must be the same on every host.
record() {
  local name="$1" file="$2"
  if [[ -n "$digests" ]]; then
    echo "$name $(sha256sum < "$file" | cut -d' ' -f1)" >> "$work/digests"
  fi
}

# Counts the rows of score output, after its header rows, in which some
# category scores at least 128.
flagged_rows() {
  awk -F'\t' '!/^#/ { for (i = 1; i < NF; i++) if ($i >= 128) { n++; break } } END { print n + 0 }' "$1"
}

# Counts the lines that the programs read from a file: one for each line
# feed, and one for trailing bytes without one.
line_count() {
  python3 -I -c '
import sys
data = open(sys.argv[1], "rb").read()
print(data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0))
' "$1"
}

echo "== Toolchain"
pinned="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$repo/rust-toolchain.toml")"
in_image="$(sed -n 's|^ARG RUST_IMAGE=.*/rust:\([0-9.]*\)-.*|\1|p' "$repo/tests/docker/Dockerfile")"
if [[ -z "$pinned" || "$pinned" != "$in_image" ]]; then
  echo "FAIL: the Rust image has $in_image, but rust-toolchain.toml pins $pinned" >&2
  exit 1
fi
pass "the Rust image provides the pinned toolchain $pinned"

echo "== Build"
docker build -f "$repo/tests/docker/Dockerfile" -t "$image" "$repo"

echo "== Inputs"
# The payloads of the labeled development attacks, without their labels.
grep -v -e '^#' -e '^$' "$repo/tests/data/attacks.tsv" | cut -f2- > "$work/input_attacks"
cp "$repo/tests/data/benign.txt" "$work/input_benign"
# Bytes that no text format allows: NUL, invalid UTF-8, a lone carriage
# return, an empty line, and a line of 1 MiB. They are the same on every
# host, so that their scores can be compared. One line carries a marker
# that no output may contain, since input bytes are never echoed.
marker="S2FP-E2E-MARKER-7f3a9c"
python3 -I - "$work/input_hostile" "$marker" <<'PYTHON'
import sys

path, marker = sys.argv[1], sys.argv[2].encode()
state = 0x5EED


def random_bytes(count):
    """Returns bytes of a fixed generator, the same with every Python."""
    global state
    out = bytearray(count)
    for i in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out[i] = state >> 56
    return bytes(out)


lines = [b"", b"\r", b"\x00" * 64, b"\xc0\xaf\xed\xa0\x80\xff\xfe", b"x" + marker + b"x"]
for length in range(1, 600):
    lines.append(random_bytes(length).replace(b"\n", b" "))
lines.append(random_bytes(1 << 20).replace(b"\n", b"\t"))
with open(path, "wb") as out:
    # The last line has no line feed.
    out.write(b"\n".join(lines))
# Bytes in place of a model file, the same on every run.
with open(path + "_model", "wb") as out:
    out.write(random_bytes(4096))
PYTHON
inputs=(attacks benign hostile)
for input in "${inputs[@]}"; do
  echo "$input: $(line_count "$work/input_$input") lines"
done
attacks="$(line_count "$work/input_attacks")"
benign="$(line_count "$work/input_benign")"

echo "== Kernel self-test"
if run kernel_self_test > "$work/self_test.txt"; then
  pass "every detected backend agrees with the scalar kernel: $(paste -sd' ' "$work/self_test.txt")"
else
  fail "the kernel self-test failed"
  cat "$work/self_test.txt" >&2
fi

echo "== Dependent crate with a model trained by its build script"
for input in "${inputs[@]}"; do
  out="$work/downstream_$input.txt"
  if ! run s2fp-downstream < "$work/input_$input" > "$out"; then
    fail "s2fp-downstream failed on $input input"
    continue
  fi
  record "downstream-$input" "$out"
  expected="$(line_count "$work/input_$input")"
  rows="$(wc -l < "$out")"
  if [[ "$rows" -ne "$expected" ]]; then
    fail "s2fp-downstream wrote $rows rows for $expected lines of $input input"
  fi
done
flagged_attacks="$(grep -cvx benign "$work/downstream_attacks.txt" || true)"
flagged_benign="$(grep -cvx benign "$work/downstream_benign.txt" || true)"
# The limits of the tests of the crate: its model is weak, so they check
# that it was trained on this data, not how good it is.
summary="flags $flagged_attacks of $attacks attacks and $flagged_benign of $benign benign lines"
if ((flagged_attacks > flagged_benign && flagged_benign * 10 <= benign)); then
  pass "s2fp-downstream $summary"
else
  fail "s2fp-downstream $summary"
fi
expect_status 2 "s2fp-downstream with an argument" s2fp-downstream unexpected

echo "== Embedded models"
for model in compact default large strict; do
  for input in "${inputs[@]}"; do
    out="$work/score_${model}_$input.txt"
    if ! run score "$model" < "$work/input_$input" > "$out"; then
      fail "score $model failed on $input input"
      continue
    fi
    expected="$(line_count "$work/input_$input")"
    rows="$(($(wc -l < "$out") - 2))"
    if [[ "$rows" -ne "$expected" ]] || ! head -n 1 "$out" | grep -q '^# backend: '; then
      fail "score $model wrote $rows rows for $expected lines of $input input"
    fi
    # The backend differs between hosts, and the scores must not.
    tail -n +2 "$out" > "$out.scores"
    record "score-$model-$input" "$out.scores"
  done
  flagged_attacks="$(flagged_rows "$work/score_${model}_attacks.txt")"
  flagged_benign="$(flagged_rows "$work/score_${model}_benign.txt")"
  # Loose limits. `tests/model_quality.rs` checks the quality of each model.
  summary="flags $flagged_attacks of $attacks attacks and $flagged_benign of $benign benign lines at 128"
  if ((flagged_attacks * 10 >= attacks * 8 && flagged_benign * 10 <= benign)); then
    pass "$model $summary"
  else
    fail "$model $summary"
  fi
done

echo "== Presets, synchronous and on Tokio"
# `detect_async` decides on a Tokio runtime, long lines on its blocking
# pool, and must print what `detect` prints. The hostile input has a line
# of 1 MiB, which goes through the pool.
for preset in fast balanced thorough strict; do
  for input in "${inputs[@]}"; do
    out="$work/detect_${preset}_$input.txt"
    if ! run detect "$preset" < "$work/input_$input" > "$out" 2> /dev/null; then
      fail "detect $preset failed on $input input"
      continue
    fi
    if ! run detect_async "$preset" < "$work/input_$input" > "$out.async" 2> /dev/null; then
      fail "detect_async $preset failed on $input input"
      continue
    fi
    expected="$(line_count "$work/input_$input")"
    rows="$(wc -l < "$out")"
    if [[ "$rows" -ne "$expected" ]]; then
      fail "detect $preset wrote $rows rows for $expected lines of $input input"
    elif ! cmp -s "$out" "$out.async"; then
      fail "detect_async $preset differs from detect on $input input"
    fi
    record "detect-$preset-$input" "$out"
  done
  pass "detect_async decides like detect with the $preset preset"
done
expect_status 2 "detect_async with an unknown preset" detect_async unknown
expect_status 2 "detect_async with two arguments" detect_async strict extra

echo "== Training from the command line"
# The settings of the build script of `tests/downstream/`. Training with
# one thread and with four must give the same model bytes. The model then
# scores the development attacks.
settings="--hidden 16 --epochs 40 --quantized-epochs 5 --batch-size 16 --learning-rate 0.01 --input-dropout 0"
if run sh -c "
  set -e
  train --benign $data/benign.txt --labeled $data/attacks.tsv $settings --threads 1 --out /tmp/one.s2fp 2> /dev/null
  train --benign $data/benign.txt --labeled $data/attacks.tsv $settings --threads 4 --out /tmp/four.s2fp 2> /dev/null
  cmp /tmp/one.s2fp /tmp/four.s2fp
  score /tmp/one.s2fp > /tmp/scores.txt
  tail -n +2 /tmp/scores.txt
  echo \"# model \$(sha256sum < /tmp/one.s2fp | cut -d' ' -f1)\"
" < "$work/input_attacks" > "$work/trained.txt"; then
  record "trained" "$work/trained.txt"
  flagged="$(flagged_rows "$work/trained.txt")"
  # The column header and the model digest, besides one row a line.
  rows="$(($(wc -l < "$work/trained.txt") - 2))"
  if [[ "$rows" -ne "$attacks" ]]; then
    fail "the trained model wrote $rows rows for $attacks lines of attacks input"
  elif ((flagged * 2 >= attacks)); then
    pass "the trained model is the same with 1 and 4 threads and flags $flagged of $attacks attacks"
  else
    fail "the trained model flags only $flagged of $attacks attacks"
  fi
else
  fail "training from the command line failed or depended on the thread count"
fi

if run sh -c "
  set -e
  train --base compact --benign $data/benign.txt --labeled $data/attacks.tsv --epochs 2 --quantized-epochs 1 --out /tmp/tuned.s2fp 2> /dev/null
  score /tmp/tuned.s2fp < /dev/null > /dev/null
  sha256sum < /tmp/tuned.s2fp
" > "$work/tuned.txt"; then
  record "fine-tuned" "$work/tuned.txt"
  pass "fine-tuning the compact model gives a model that loads"
else
  fail "fine-tuning the compact model failed"
fi

echo "== Model versions"
# Every embedded model and a model trained here load, and their versions
# and checksums are the same on every host.
if run sh -c "
  set -e
  train --benign $data/benign.txt --labeled $data/attacks.tsv --hidden 4 --epochs 2 --quantized-epochs 1 --out /tmp/small.s2fp 2> /dev/null
  model_info compact default large strict /tmp/small.s2fp
" > "$work/versions.txt"; then
  record "model-info" "$work/versions.txt"
  if [[ "$(grep -c ': layout [0-9][0-9.]*, library .*, layers ' "$work/versions.txt")" -eq 5 ]]; then
    pass "model_info loads every embedded model and a trained one"
  else
    fail "model_info printed unexpected lines"
    cat "$work/versions.txt" >&2
  fi
else
  fail "a model did not load in model_info"
fi

echo "== Failures"
expect_status 2 "train with an unknown option" train --unknown
expect_status 2 "train without --out" train --benign "$data/benign.txt"
expect_status 1 "train with a missing file" train --benign /tmp/missing --out /tmp/model.s2fp
expect_status 2 "train with a library requirement that this library does not match" \
  train --benign "$data/benign.txt" --library-requirement '>=999' --out /tmp/model.s2fp
expect_status 2 "model_info without a model" model_info
expect_status 1 "model_info with a text file as the model" model_info "$data/benign.txt"
expect_status 2 "score with two arguments" score default extra
expect_status 1 "score with a missing model file" score /tmp/missing
stdin="$work/input_hostile_model" expect_status 1 "score with a model file of random bytes" \
  sh -c 'cat > /tmp/bad.s2fp && score /tmp/bad.s2fp'
expect_status 1 "score with a text file as the model" score "$data/benign.txt"
expect_status 1 "score with a truncated model file" sh -c "
  train --benign $data/benign.txt --labeled $data/attacks.tsv $settings --out /tmp/model.s2fp 2> /dev/null
  head -c 1000 /tmp/model.s2fp > /tmp/truncated.s2fp
  score /tmp/truncated.s2fp
"

if grep -qsF "$marker" "$work"/downstream_* "$work"/score_* "$work"/detect_* "$work/trained.txt"; then
  fail "an output contains input bytes"
else
  pass "no output contains input bytes"
fi

if [[ -n "$digests" ]]; then
  if [[ -f "$work/digests" ]]; then
    cp "$work/digests" "$digests"
    echo "wrote $(wc -l < "$digests") digests to $digests"
  else
    fail "no digest was recorded"
  fi
fi

if ((failures > 0)); then
  echo "$failures end-to-end checks failed" >&2
  exit 1
fi
echo "every end-to-end check passed"
