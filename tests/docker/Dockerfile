# Builds the dependent crate in `tests/downstream/`, which trains its model
# in its build script with the `build` feature, and the command-line
# examples, from a copy of the working tree. The final image holds only
# the programs and the development data, and `tests/docker/e2e.sh` runs it
# without a network, as a user without privileges, on a read-only file
# system.
#
# The Rust image must provide the toolchain that `rust-toolchain.toml` pins,
# so update both together. `tests/docker/e2e.sh` compares the tag with the
# pin, and the build fails when the image lacks the pinned toolchain, since
# it has no network and `rustup` may not install one. Both images are
# pinned by the digest of their multi-platform index.
ARG RUST_IMAGE=docker.io/library/rust:1.100.0-slim-trixie@sha256:3b199fd6db7d709274df56565b8f4003f9e30d7bfcff9d53148c86053b7855e2
ARG RUNTIME_IMAGE=docker.io/library/debian:trixie-slim@sha256:8aa5a7bf5a1955bddbe7d74fcdd0b91ab30e74362a8b73af672b0481b35b44a0

FROM ${RUST_IMAGE} AS build
# The crate has no dependencies, so the build needs no network.
ENV CARGO_TERM_COLOR=never \
    CARGO_NET_OFFLINE=true \
    RUSTUP_AUTO_INSTALL=0
WORKDIR /src
COPY . .
# The image installs the toolchain with its minimal profile. Naming the
# channel of `rust-toolchain.toml` keeps `rustup` from downloading the
# components that the file adds for development.
RUN --network=none RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)" \
    && export RUSTUP_TOOLCHAIN \
    && rustc --version --verbose \
    && cargo build --release --locked --manifest-path tests/downstream/Cargo.toml --bin s2fp-downstream \
    && cargo build --release --locked --features train \
        --example score --example train --example kernel_self_test \
    && mkdir -p /out/bin /out/data \
    && cp tests/downstream/target/release/s2fp-downstream \
        target/release/examples/score \
        target/release/examples/train \
        target/release/examples/kernel_self_test \
        /out/bin/ \
    && cp tests/data/benign.txt tests/data/attacks.tsv /out/data/

FROM ${RUNTIME_IMAGE} AS runtime
COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/data/ /usr/local/share/string2fingerprint/
# A numeric user, so that `docker run` needs no passwd entry for it.
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT []
CMD ["kernel_self_test"]
