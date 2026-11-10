//! Fuzzing on the null backend: arbitrary bytes → queue writes and submissions of buffer copies and
//! clears checked against a model of the buffers' contents (DECISIONS #21).
#![no_main]

use libfuzzer_sys::fuzz_target;

// The same oracle as in the `pragma_rhi` property tests (`tests/oracle.rs`).
#[path = "../../crates/pragma_rhi/tests/common/mod.rs"]
mod common;

fuzz_target!(|data: &[u8]| common::transfers::check(common::transfers::decode(data)));
