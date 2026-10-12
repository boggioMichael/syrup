#!/usr/bin/env bash
# Build the Syrup side of the comparison for two commits of this repository:
# the library before the optimisations and after them, both with the
# harness of the current checkout (examples/vs_opencv.rs), so the only
# difference between the two binaries is the library. The before tree has
# no CascadeOptions::min_deviation, so the harness line passing it is
# dropped there. Also builds the YuNet side (F2) from the after commit.
#
#     bench/vs-opencv/build.sh BEFORE_COMMIT AFTER_COMMIT OUT_DIR
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
before=${1:?before commit}
after=${2:?after commit}
out=$(mkdir -p "${3:?output directory}" && cd "$3" && pwd)

tree_for() { # commit name
    local tree="$out/src-$2"
    if [ -d "$tree" ]; then git -C "$repo" worktree remove --force "$tree" 2>/dev/null || rm -rf "$tree"; fi
    git -C "$repo" worktree add -f --detach "$tree" "$1" >/dev/null
    if [ -f "$repo/.cargo/config.toml" ]; then # a local vendored-crates setup, if any
        mkdir -p "$tree/.cargo" && cp "$repo/.cargo/config.toml" "$tree/.cargo/"
    fi
    cp "$repo/examples/vs_opencv.rs" "$tree/examples/vs_opencv.rs"
    if ! grep -q "min_deviation" "$tree/src/cascade.rs"; then
        sed -i '/min_deviation: a.min_deviation,/d' "$tree/examples/vs_opencv.rs"
    fi
    echo "$tree"
}

for side in before after; do
    commit=$([ "$side" = before ] && echo "$before" || echo "$after")
    tree=$(tree_for "$commit" "$side")
    version=$(git -C "$tree" rev-parse --short HEAD)
    (cd "$tree" && SYRUP_BENCH_VERSION="$version" CARGO_TARGET_DIR="$out/target-$side" \
        cargo build --release --example vs_opencv)
    cp "$out/target-$side/release/examples/vs_opencv" "$out/vs_opencv-$side"
    (cd "$tree" && SYRUP_BENCH_VERSION="$version" CARGO_TARGET_DIR="$out/target-$side" cargo build --release --lib)
    cp "$out/target-$side/release/libsyrup.so" "$out/libsyrup-$side.so"
done
tree="$out/src-after"
(cd "$tree" && SYRUP_BENCH_VERSION="$(git -C "$tree" rev-parse --short HEAD)" CARGO_TARGET_DIR="$out/target-after" \
    cargo build --release -p syrup-runtime --example vs_opencv_yunet)
cp "$out/target-after/release/examples/vs_opencv_yunet" "$out/vs_opencv_yunet"
sha256sum "$out"/vs_opencv-before "$out"/vs_opencv-after "$out"/vs_opencv_yunet "$out"/libsyrup-*.so
