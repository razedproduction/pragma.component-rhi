//! Property test: buffer handles against a naive `HashMap` oracle (roadmap M1).
//!
//! Random create / destroy / query sequences, including stale handles and double destroys, on a
//! null device must agree with a map from handle to size at every step.
#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::collections::HashMap;

use pragma_rhi::{
    Buffer, BufferDescriptor, BufferUsages, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    NullConfig, ResourceKind, RhiError,
};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Create(u64),
    /// Destroy the n-th handle ever created (live or not).
    Destroy(usize),
    /// Query the n-th handle ever created (live or not).
    Query(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1_u64..=4096).prop_map(Op::Create),
        any::<usize>().prop_map(Op::Destroy),
        any::<usize>().prop_map(Op::Query),
    ]
}

fn null_device(budget: u64) -> Device {
    let desc = InstanceDescriptor {
        null: NullConfig {
            memory_budget: budget,
            ..NullConfig::default()
        },
        ..InstanceDescriptor::null()
    };
    let adapter = Instance::new(&desc).adapters().remove(0);
    adapter
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn buffers_agree_with_the_oracle(ops in proptest::collection::vec(op(), 1..200)) {
        const BUDGET: u64 = 64 * 1024;
        let mut device = null_device(BUDGET);
        let mut oracle: HashMap<Buffer, u64> = HashMap::new();
        let mut ever: Vec<Buffer> = Vec::new();

        for op in ops {
            match op {
                Op::Create(size) => {
                    let used: u64 = oracle.values().sum();
                    let result = device.create_buffer(&BufferDescriptor {
                        label: None,
                        size,
                        usage: BufferUsages::STORAGE,
                    });
                    if used + size > BUDGET {
                        let oom = matches!(result, Err(RhiError::OutOfMemory { bytes, .. }) if bytes == size);
                        prop_assert!(oom, "over budget is OutOfMemory");
                    } else {
                        let buffer = result?;
                        prop_assert!(!ever.contains(&buffer), "a handle is never handed out twice");
                        oracle.insert(buffer, size);
                        ever.push(buffer);
                    }
                }
                Op::Destroy(n) if !ever.is_empty() => {
                    let buffer = ever[n % ever.len()];
                    let result = device.destroy_buffer(buffer);
                    if oracle.remove(&buffer).is_some() {
                        prop_assert_eq!(result, Ok(()));
                    } else {
                        let stale = matches!(result, Err(RhiError::StaleHandle { .. }));
                        prop_assert!(stale, "a second destroy is StaleHandle");
                    }
                }
                Op::Query(n) if !ever.is_empty() => {
                    let buffer = ever[n % ever.len()];
                    let got = device.buffer_info(buffer).map(|info| info.size);
                    match oracle.get(&buffer) {
                        Some(size) => prop_assert_eq!(got, Ok(*size)),
                        None => {
                            let stale = matches!(got, Err(RhiError::StaleHandle { .. }));
                            prop_assert!(stale, "a destroyed handle is StaleHandle");
                        }
                    }
                }
                Op::Destroy(_) | Op::Query(_) => {}
            }
            prop_assert_eq!(device.live(ResourceKind::Buffer), oracle.len());
            prop_assert_eq!(device.allocated_bytes(), Some(oracle.values().sum()));
        }
    }
}
