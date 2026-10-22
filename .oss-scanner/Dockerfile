# Build image for the Anthropic OSS Scanner (https://github.com/anthropics/oss-scanner).
# The scanner builds this file with the repository as the build context, then runs
# the image without network access. Everything the tests and fuzz targets need is
# fetched here.
FROM rust:1-bookworm

ENV DEBIAN_FRONTEND=noninteractive
# g++ and make build the C++ LIBSVM reference in vendor/libsvm. The rust image ships
# them too; they are listed here so the dependency is explicit.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 clang g++ make \
    && rm -rf /var/lib/apt/lists/*

# Number of parallel compile jobs. The scanner's build machine has 16 CPUs.
ARG JOBS=4
ENV CARGO_BUILD_JOBS=${JOBS} \
    RUSTFLAGS="-C force-frame-pointers=yes"

COPY . /src
WORKDIR /src

# rust-toolchain.toml pins the stable toolchain; rustup installs it on first use.
RUN cargo fetch --locked \
    && cargo build --locked --workspace --all-targets --all-features \
    && cargo test --locked --workspace --all-features --no-run

# Fuzz targets (crates/libsvm/fuzz) need the nightly pinned in rust-toolchain-fuzz.toml.
RUN NIGHTLY="$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain-fuzz.toml)" \
    && rustup toolchain install "$NIGHTLY" --profile minimal \
    && cargo +"$NIGHTLY" install cargo-fuzz --locked \
    && cd crates/libsvm \
    && cargo +"$NIGHTLY" fuzz build

# The C++ reference tools (svm-train, svm-predict, svm-scale). The comparison test in
# crates/libsvm/src/predict.rs returns early when vendor/libsvm/svm-predict is missing.
RUN make -C vendor/libsvm

# Run the tests once so a broken image is visible in the build log. A failure does not stop the build.
RUN cargo test --locked --workspace --all-features || echo "WARNING: cargo test reported failures"

# From here on there is no network.
ENV CARGO_NET_OFFLINE=true
