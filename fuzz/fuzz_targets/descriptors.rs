//! Fuzzing on the null backend: arbitrary bytes → arbitrary buffer, texture, view and sampler
//! descriptors checked against the modeled validation rules; a failed creation must change nothing
//! (DECISIONS #21).
#![no_main]

use libfuzzer_sys::fuzz_target;

// The same oracle as in the `pragma_rhi` property tests (`tests/oracle.rs`).
#[path = "../../crates/pragma_rhi/tests/common/mod.rs"]
mod common;

fuzz_target!(|data: &[u8]| common::descriptors::check(common::descriptors::decode(data)));
