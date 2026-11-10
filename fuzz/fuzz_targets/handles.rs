//! Fuzzing on the null backend: arbitrary bytes → buffers, textures and views created, destroyed
//! and queried, with submissions, polls and frames, checked after every operation against the model
//! of handles, memory and deferred destruction (DECISIONS #21).
#![no_main]

use libfuzzer_sys::fuzz_target;

// The same oracle as in the `pragma_rhi` property tests (`tests/oracle.rs`).
#[path = "../../crates/pragma_rhi/tests/common/mod.rs"]
mod common;

fuzz_target!(|data: &[u8]| common::handles::check(common::handles::decode(data)));
