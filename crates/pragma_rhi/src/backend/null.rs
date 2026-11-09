//! The null backend: no GPU, deterministic. It keeps no resource data; it accounts device
//! memory against a fixed budget, so out-of-memory paths can be tested on any machine.

use crate::RhiError;

#[derive(Debug)]
pub(crate) struct NullDevice {
    budget: u64,
    allocated: u64,
    lost: bool,
}

impl NullDevice {
    pub(crate) const fn new(budget: u64) -> Self {
        Self {
            budget,
            allocated: 0,
            lost: false,
        }
    }

    pub(crate) const fn is_lost(&self) -> bool {
        self.lost
    }

    pub(crate) fn lose(&mut self) {
        self.lost = true;
    }

    pub(crate) const fn allocated(&self) -> u64 {
        self.allocated
    }

    pub(crate) fn allocate(&mut self, bytes: u64) -> Result<(), RhiError> {
        match self.allocated.checked_add(bytes) {
            Some(total) if total <= self.budget => {
                self.allocated = total;
                Ok(())
            }
            _ => Err(RhiError::OutOfMemory { bytes }),
        }
    }

    pub(crate) fn release(&mut self, bytes: u64) {
        self.allocated = self.allocated.saturating_sub(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::NullDevice;
    use crate::RhiError;

    #[test]
    fn budget_is_enforced() {
        let mut device = NullDevice::new(100);
        assert_eq!(device.allocate(60), Ok(()));
        assert_eq!(
            device.allocate(41),
            Err(RhiError::OutOfMemory { bytes: 41 })
        );
        assert_eq!(
            device.allocate(u64::MAX),
            Err(RhiError::OutOfMemory { bytes: u64::MAX })
        );
        device.release(60);
        assert_eq!(device.allocate(100), Ok(()));
        assert_eq!(device.allocated(), 100);
    }
}
