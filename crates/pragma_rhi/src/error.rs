//! The crate's error type.

/// An error of an RHI operation.
///
/// New variants are added as the API grows (`#[non_exhaustive]`), so match with a `_` arm.
///
/// ```
/// use pragma_rhi::RhiError;
///
/// fn describe(err: &RhiError) -> String {
///     match err {
///         RhiError::DeviceLost => "recreate the device".to_owned(),
///         RhiError::OutOfMemory { bytes, .. } => format!("allocation of {bytes} bytes failed"),
///         _ => err.to_string(),
///     }
/// }
/// # let _ = describe;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RhiError {
    /// An allocation failed, in host memory (`try_reserve`) or in device memory.
    #[error("out of memory while allocating {bytes} bytes")]
    #[non_exhaustive]
    OutOfMemory {
        /// The size of the allocation that failed.
        bytes: u64,
    },
    /// The device was lost (driver reset, removal); every handle of it is now unusable and the
    /// engine has to create a new device.
    #[error("the device was lost")]
    DeviceLost,
}

#[cfg(test)]
mod tests {
    use super::RhiError;

    #[test]
    fn messages_name_the_failure() {
        assert_eq!(
            RhiError::OutOfMemory { bytes: 64 }.to_string(),
            "out of memory while allocating 64 bytes"
        );
        assert_eq!(RhiError::DeviceLost.to_string(), "the device was lost");
    }
}
