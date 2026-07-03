#!/bin/bash

set -ex
cargo build --release
diff -q testdata/stress-1000-1.txt <(./target/release/stress_dump 1000 1)
diff -q testdata/stress-5000-42.txt <(./target/release/stress_dump 5000 42)
