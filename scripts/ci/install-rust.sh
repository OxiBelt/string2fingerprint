#!/usr/bin/env bash
# Adapted from qquic scripts/ci/install-rust.sh (Apache-2.0).
# Source revision: 59f42d52ece49c841d9e1f027db9da578eb62e4e.
set -euo pipefail

if ! command -v rustup >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64) host=x86_64-unknown-linux-gnu ;;
    aarch64) host=aarch64-unknown-linux-gnu ;;
    riscv64) host=riscv64gc-unknown-linux-gnu ;;
    *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
  esac

  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  base_url="https://static.rust-lang.org/rustup/dist/$host"
  curl --fail --silent --show-error --location --retry 3 --proto '=https' \
    "$base_url/rustup-init" --output "$download_dir/rustup-init"
  curl --fail --silent --show-error --location --retry 3 --proto '=https' \
    "$base_url/rustup-init.sha256" --output "$download_dir/rustup-init.sha256"
  printf '%s  %s\n' "$(cut -d' ' -f1 < "$download_dir/rustup-init.sha256")" \
    "$download_dir/rustup-init" | sha256sum --check --strict
  chmod +x "$download_dir/rustup-init"
  "$download_dir/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none

  cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
  export PATH="$cargo_bin:$PATH"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    printf '%s\n' "$cargo_bin" >> "$GITHUB_PATH"
  fi
fi

# Install the channel and components defined in rust-toolchain.toml.
rustup toolchain install
rustc --version --verbose
