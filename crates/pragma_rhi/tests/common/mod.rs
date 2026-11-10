//! Oracles shared by the property tests and the fuzz targets (DECISIONS #21).
//!
//! Each module defines the operations of one property, a harness that applies them to a null
//! [`pragma_rhi::Device`] and to a naive model and panics on the first disagreement, and a decoder
//! that turns arbitrary bytes (a fuzzer's input) into the same operations the property tests
//! generate. `tests/oracle.rs` and `fuzz/fuzz_targets/*.rs` include this module through `mod` /
//! `#[path]`; not every includer uses every module.
// Test code: a panic is a test failure; not every includer uses every item.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unreachable
)]

pub mod bytes;
pub mod descriptors;
pub mod handles;
pub mod transfers;

use pragma_rhi::{Device, DeviceDescriptor, Instance, InstanceDescriptor, NullConfig};

/// A null device with the given memory budget and submission latency.
pub fn null_device(memory_budget: u64, latency: u32) -> Device {
    let desc = InstanceDescriptor {
        null: NullConfig {
            memory_budget,
            latency,
            ..NullConfig::default()
        },
        ..InstanceDescriptor::null()
    };
    Instance::new(&desc)
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}
