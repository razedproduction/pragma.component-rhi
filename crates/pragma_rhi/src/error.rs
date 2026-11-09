//! The crate's error type.

use crate::handle::ResourceKind;

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
        /// The size of the allocation that failed (`u64::MAX` when the size itself overflowed).
        bytes: u64,
    },
    /// The device was lost (driver reset, removal); every handle of it is now unusable and the
    /// engine has to create a new device.
    #[error("the device was lost")]
    DeviceLost,
    /// The handle refers to a resource that was destroyed (its slot may have been reused).
    #[error("stale {resource} handle: the resource was destroyed")]
    #[non_exhaustive]
    StaleHandle {
        /// The kind of the resource.
        resource: ResourceKind,
    },
    /// The handle was created by another device.
    #[error("the {resource} handle belongs to another device")]
    #[non_exhaustive]
    ForeignHandle {
        /// The kind of the resource.
        resource: ResourceKind,
    },
    /// A descriptor breaks a rule that does not depend on the device's limits.
    #[error("invalid {resource} descriptor: {reason}")]
    #[non_exhaustive]
    InvalidDescriptor {
        /// The kind of the resource being described.
        resource: ResourceKind,
        /// The rule that was broken.
        reason: &'static str,
    },
    /// A requested value is larger than the device's (or the adapter's) limit allows.
    #[error("limit `{limit}` exceeded: requested {requested}, the maximum is {max}")]
    #[non_exhaustive]
    LimitExceeded {
        /// The name of the limit, usually a field of [`Limits`](crate::Limits).
        limit: &'static str,
        /// The requested value.
        requested: u64,
        /// The largest allowed value.
        max: u64,
    },
    /// The resource cannot be destroyed while other resources depend on it (a texture with
    /// live views).
    #[error("the {resource} is still used by {dependents} other resources")]
    #[non_exhaustive]
    ResourceInUse {
        /// The kind of the resource.
        resource: ResourceKind,
        /// How many resources still depend on it.
        dependents: u32,
    },
    /// The backend rejected an operation that passed `pragma_rhi`'s own validation, or failed
    /// to create a device. The message comes from the backend and is not stable.
    #[error("backend error: {message}")]
    #[non_exhaustive]
    Backend {
        /// The backend's description of the failure.
        message: String,
    },
}

impl RhiError {
    pub(crate) const fn invalid(resource: ResourceKind, reason: &'static str) -> Self {
        Self::InvalidDescriptor { resource, reason }
    }

    pub(crate) fn limit(
        limit: &'static str,
        requested: impl Into<u64>,
        max: impl Into<u64>,
    ) -> Self {
        Self::LimitExceeded {
            limit,
            requested: requested.into(),
            max: max.into(),
        }
    }

    pub(crate) fn backend(message: impl Into<String>) -> Self {
        Self::Backend {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RhiError;
    use crate::ResourceKind;

    #[test]
    fn messages_name_the_failure() {
        assert_eq!(
            RhiError::OutOfMemory { bytes: 64 }.to_string(),
            "out of memory while allocating 64 bytes"
        );
        assert_eq!(RhiError::DeviceLost.to_string(), "the device was lost");
        assert_eq!(
            RhiError::StaleHandle {
                resource: ResourceKind::TextureView
            }
            .to_string(),
            "stale texture view handle: the resource was destroyed"
        );
        assert_eq!(
            RhiError::limit("max_buffer_size", 10_u64, 4_u64).to_string(),
            "limit `max_buffer_size` exceeded: requested 10, the maximum is 4"
        );
    }
}
