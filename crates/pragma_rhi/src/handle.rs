//! Generational resource handles and the per-device resource table (DECISIONS.md #9).

use core::fmt;
use core::marker::PhantomData;

use crate::RhiError;

/// The kind of a resource, as reported in errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ResourceKind {
    /// A [`Buffer`].
    Buffer,
    /// A [`Texture`].
    Texture,
    /// A [`TextureView`].
    TextureView,
    /// A [`Sampler`].
    Sampler,
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::TextureView => "texture view",
            Self::Sampler => "sampler",
        })
    }
}

/// Device + slot index + generation. A handle is valid while its slot holds a value created at
/// the same generation; removing the value bumps the generation, so every copy goes stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RawHandle {
    pub(crate) device: u32,
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

pub(crate) trait Handle: Copy {
    const KIND: ResourceKind;
    fn raw(self) -> RawHandle;
    fn from_raw(raw: RawHandle) -> Self;
}

macro_rules! handle {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        ///
        /// A `Copy` generational id (12 bytes), cheap to store in ECS components. It does not
        /// own the resource: destruction is explicit, through the [`Device`](crate::Device) that
        /// created it, and using the handle afterwards returns
        /// [`RhiError::StaleHandle`](crate::RhiError::StaleHandle).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(RawHandle);

        impl Handle for $name {
            const KIND: ResourceKind = ResourceKind::$name;
            fn raw(self) -> RawHandle {
                self.0
            }
            fn from_raw(raw: RawHandle) -> Self {
                Self(raw)
            }
        }
    };
}

handle!(
    /// A GPU buffer.
    Buffer
);
handle!(
    /// A GPU texture.
    Texture
);
handle!(
    /// A view into a [`Texture`]; the texture cannot be destroyed while it has views.
    TextureView
);
handle!(
    /// A texture sampler.
    Sampler
);

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// The slots of one resource kind on one device.
#[derive(Debug)]
pub(crate) struct Table<H, T> {
    device: u32,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    live: usize,
    _handle: PhantomData<fn() -> H>,
}

impl<H: Handle, T> Table<H, T> {
    pub(crate) const fn new(device: u32) -> Self {
        Self {
            device,
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            _handle: PhantomData,
        }
    }

    /// The number of live values.
    pub(crate) const fn len(&self) -> usize {
        self.live
    }

    /// Makes sure that the next [`Self::insert`] cannot fail, so callers reserve before they
    /// create a backend object and never have to roll one back.
    pub(crate) fn reserve(&mut self) -> Result<(), RhiError> {
        if !self.free.is_empty() {
            return Ok(());
        }
        if u32::try_from(self.slots.len()).is_err() {
            return Err(RhiError::limit(
                "resource table slots",
                self.slots.len() as u64,
                u64::from(u32::MAX),
            ));
        }
        self.slots.try_reserve(1).map_err(|_| oom::<Slot<T>>())
    }

    /// Inserts a value. Infallible after a successful [`Self::reserve`].
    pub(crate) fn insert(&mut self, value: T) -> Result<H, RhiError> {
        self.reserve()?;
        let raw = if let Some(index) = self.free.pop() {
            let slot = self
                .slots
                .get_mut(index as usize)
                .ok_or_else(oom::<Slot<T>>)?;
            slot.value = Some(value);
            RawHandle {
                device: self.device,
                index,
                generation: slot.generation,
            }
        } else {
            let index = u32::try_from(self.slots.len()).map_err(|_| oom::<Slot<T>>())?;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            RawHandle {
                device: self.device,
                index,
                generation: 0,
            }
        };
        self.live = self.live.saturating_add(1);
        Ok(H::from_raw(raw))
    }

    fn slot(&self, handle: H) -> Result<&Slot<T>, RhiError> {
        let raw = handle.raw();
        if raw.device != self.device {
            return Err(RhiError::ForeignHandle { resource: H::KIND });
        }
        match self.slots.get(raw.index as usize) {
            Some(slot) if slot.generation == raw.generation && slot.value.is_some() => Ok(slot),
            _ => Err(RhiError::StaleHandle { resource: H::KIND }),
        }
    }

    fn slot_mut(&mut self, handle: H) -> Result<&mut Slot<T>, RhiError> {
        let raw = handle.raw();
        if raw.device != self.device {
            return Err(RhiError::ForeignHandle { resource: H::KIND });
        }
        match self.slots.get_mut(raw.index as usize) {
            Some(slot) if slot.generation == raw.generation && slot.value.is_some() => Ok(slot),
            _ => Err(RhiError::StaleHandle { resource: H::KIND }),
        }
    }

    pub(crate) fn get(&self, handle: H) -> Result<&T, RhiError> {
        self.slot(handle)?
            .value
            .as_ref()
            .ok_or(RhiError::StaleHandle { resource: H::KIND })
    }

    pub(crate) fn get_mut(&mut self, handle: H) -> Result<&mut T, RhiError> {
        self.slot_mut(handle)?
            .value
            .as_mut()
            .ok_or(RhiError::StaleHandle { resource: H::KIND })
    }

    /// Removes the value; the handle and every copy of it go stale. A slot whose generation
    /// would overflow is retired instead of reused, so a stale handle never becomes valid again.
    pub(crate) fn remove(&mut self, handle: H) -> Result<T, RhiError> {
        self.slot(handle)?;
        self.free.try_reserve(1).map_err(|_| oom::<u32>())?;
        let slot = self.slot_mut(handle)?;
        let value = slot
            .value
            .take()
            .ok_or(RhiError::StaleHandle { resource: H::KIND })?;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(handle.raw().index);
        }
        self.live = self.live.saturating_sub(1);
        Ok(value)
    }

    #[cfg(test)]
    fn set_generation_for_test(&mut self, index: u32, generation: u32) {
        if let Some(slot) = self.slots.get_mut(index as usize) {
            slot.generation = generation;
        }
    }
}

fn oom<T>() -> RhiError {
    RhiError::OutOfMemory {
        bytes: core::mem::size_of::<T>() as u64,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Buffer, Handle, RawHandle, Table};
    use crate::{ResourceKind, RhiError};

    #[test]
    fn insert_get_remove() {
        let mut table: Table<Buffer, u32> = Table::new(1);
        let a = table.insert(10).unwrap();
        let b = table.insert(20).unwrap();
        assert_eq!(table.get(a), Ok(&10));
        assert_eq!(table.get(b), Ok(&20));
        assert_eq!(table.remove(a), Ok(10));
        assert_eq!(
            table.get(a),
            Err(RhiError::StaleHandle {
                resource: ResourceKind::Buffer
            })
        );
        assert!(table.remove(a).is_err(), "double destroy is an error");
        // The slot is reused with a new generation; the old handle stays stale.
        let c = table.insert(30).unwrap();
        assert_eq!(c.raw().index, a.raw().index);
        assert_ne!(c, a);
        assert!(table.get(a).is_err());
        assert_eq!(table.get(c), Ok(&30));
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn foreign_handle_is_rejected() {
        let mut one: Table<Buffer, u32> = Table::new(1);
        let two: Table<Buffer, u32> = Table::new(2);
        let h = one.insert(1).unwrap();
        assert_eq!(
            two.get(h),
            Err(RhiError::ForeignHandle {
                resource: ResourceKind::Buffer
            })
        );
    }

    #[test]
    fn out_of_range_handle_is_stale() {
        let table: Table<Buffer, u32> = Table::new(1);
        let h = Buffer::from_raw(RawHandle {
            device: 1,
            index: 7,
            generation: 0,
        });
        assert!(matches!(table.get(h), Err(RhiError::StaleHandle { .. })));
    }

    #[test]
    fn exhausted_slot_is_retired() {
        let mut table: Table<Buffer, u32> = Table::new(1);
        let first = table.insert(1).unwrap();
        table.set_generation_for_test(first.raw().index, u32::MAX);
        let h = Buffer::from_raw(RawHandle {
            generation: u32::MAX,
            ..first.raw()
        });
        assert_eq!(table.remove(h), Ok(1));
        assert!(table.get(h).is_err());
        let next = table.insert(2).unwrap();
        assert_ne!(
            next.raw().index,
            h.raw().index,
            "a retired slot is never reused"
        );
    }
}
