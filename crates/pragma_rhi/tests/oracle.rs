//! Property tests against the oracles of `tests/common` (roadmap M5, DECISIONS #21).
//!
//! The same oracles back the fuzz targets in `fuzz/`; here proptest feeds them random bytes
//! through the very decoders the fuzzers use, so a fuzz finding reproduces as a test input and
//! Windows (where the fuzz jobs do not run) checks the same properties.

mod common;

use proptest::prelude::*;

fn input() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..2048)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Handles, memory accounting, deferred destruction and fences agree with the model after
    /// every operation.
    #[test]
    fn handles_match_the_model(data in input()) {
        common::handles::check(common::handles::decode(&data));
    }

    /// Descriptor validation agrees with the modeled rules; a failure changes nothing.
    #[test]
    fn descriptors_match_the_rules(data in input()) {
        common::descriptors::check(common::descriptors::decode(&data));
    }

    /// Queue writes, copies and clears produce exactly the modeled contents; a broken
    /// submission changes nothing.
    #[test]
    fn transfers_match_the_model(data in input()) {
        common::transfers::check(common::transfers::decode(&data));
    }
}
