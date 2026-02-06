//! Multi-band, multi-channel audio block.
//!
//! Ported from `modules/audio_processing/aec3/block.h`.

use crate::common::BLOCK_SIZE;

/// Contains one or more channels of 4 ms of audio data.
///
/// The audio is split into one or more frequency bands, each with a sampling
/// rate of 16 kHz. Each band/channel combination holds `BLOCK_SIZE` (64)
/// samples.
#[derive(Clone)]
pub(crate) struct Block {
    num_bands: usize,
    num_channels: usize,
    data: Vec<f32>,
}

impl Block {
    pub(crate) fn new(num_bands: usize, num_channels: usize) -> Self {
        Self {
            num_bands,
            num_channels,
            data: vec![0.0; num_bands * num_channels * BLOCK_SIZE],
        }
    }

    pub(crate) fn new_with_value(num_bands: usize, num_channels: usize, value: f32) -> Self {
        Self {
            num_bands,
            num_channels,
            data: vec![value; num_bands * num_channels * BLOCK_SIZE],
        }
    }

    pub(crate) fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub(crate) fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Modifies the number of channels and zeros all samples.
    pub(crate) fn set_num_channels(&mut self, num_channels: usize) {
        self.num_channels = num_channels;
        self.data
            .resize(self.num_bands * self.num_channels * BLOCK_SIZE, 0.0);
        self.data.fill(0.0);
    }

    /// Returns a slice of `BLOCK_SIZE` samples for the given band and channel.
    pub(crate) fn view(&self, band: usize, channel: usize) -> &[f32] {
        let idx = self.get_index(band, channel);
        &self.data[idx..idx + BLOCK_SIZE]
    }

    /// Returns a mutable slice of `BLOCK_SIZE` samples for the given band and channel.
    pub(crate) fn view_mut(&mut self, band: usize, channel: usize) -> &mut [f32] {
        let idx = self.get_index(band, channel);
        &mut self.data[idx..idx + BLOCK_SIZE]
    }

    /// Swaps audio data with another block.
    pub(crate) fn swap(&mut self, other: &mut Block) {
        std::mem::swap(&mut self.num_bands, &mut other.num_bands);
        std::mem::swap(&mut self.num_channels, &mut other.num_channels);
        std::mem::swap(&mut self.data, &mut other.data);
    }

    fn get_index(&self, band: usize, channel: usize) -> usize {
        (band * self.num_channels + channel) * BLOCK_SIZE
    }
}
