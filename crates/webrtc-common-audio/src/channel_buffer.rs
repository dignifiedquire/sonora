//! Multi-channel, multi-band audio buffer matching WebRTC's `ChannelBuffer`.
//!
//! The buffer stores audio data in a single contiguous allocation, laid out as:
//!
//! ```text
//! [ band0_ch0 | band1_ch0 | band0_ch1 | band1_ch1 ]
//! ```
//!
//! Two indexing schemes are provided:
//! - **By band then channel:** `channel(band, ch)` — used for per-band processing
//! - **By channel then band:** `band(ch, band)` — used for per-channel frequency access

use derive_more::Debug;

/// Multi-channel, optionally multi-band audio buffer.
///
/// The number of bands is determined by sample rate:
/// - 8 kHz / 16 kHz: 1 band
/// - 32 kHz: 2 bands
/// - 48 kHz: 3 bands
#[derive(Debug)]
pub struct ChannelBuffer<T> {
    #[debug(skip)]
    data: Vec<T>,
    num_frames: usize,
    num_frames_per_band: usize,
    num_allocated_channels: usize,
    /// User-visible channel count (can be reduced via [`set_num_channels`]).
    num_channels: usize,
    num_bands: usize,
}

impl<T: Clone + Default> ChannelBuffer<T> {
    /// Create a new zero-initialized buffer.
    ///
    /// `num_frames` must be divisible by `num_bands`.
    pub fn new(num_frames: usize, num_channels: usize, num_bands: usize) -> Self {
        assert!(num_bands > 0, "num_bands must be > 0");
        assert!(num_channels > 0, "num_channels must be > 0");
        assert!(
            num_frames.is_multiple_of(num_bands),
            "num_frames ({num_frames}) must be divisible by num_bands ({num_bands})"
        );
        Self {
            data: vec![T::default(); num_frames * num_channels],
            num_frames,
            num_frames_per_band: num_frames / num_bands,
            num_allocated_channels: num_channels,
            num_channels,
            num_bands,
        }
    }

    /// Create a single-band buffer (the common case).
    pub fn new_single_band(num_frames: usize, num_channels: usize) -> Self {
        Self::new(num_frames, num_channels, 1)
    }
}

impl<T> ChannelBuffer<T> {
    /// Total number of frames across all bands.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Number of frames in each band.
    #[inline]
    pub fn num_frames_per_band(&self) -> usize {
        self.num_frames_per_band
    }

    /// Number of visible channels (may be less than allocated).
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Number of frequency bands.
    #[inline]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Total number of elements in the buffer.
    #[inline]
    pub fn size(&self) -> usize {
        self.num_frames * self.num_allocated_channels
    }

    /// Set the user-visible number of channels.
    ///
    /// Must be <= the allocated channel count.
    pub fn set_num_channels(&mut self, num_channels: usize) {
        assert!(
            num_channels <= self.num_allocated_channels,
            "num_channels ({num_channels}) exceeds allocated ({0})",
            self.num_allocated_channels
        );
        self.num_channels = num_channels;
    }

    /// Returns the offset into `data` for a given channel and band.
    #[inline]
    fn offset(&self, channel: usize, band: usize) -> usize {
        channel * self.num_frames + band * self.num_frames_per_band
    }

    /// Get a slice for a specific channel and band.
    ///
    /// With `band = 0` and single-band buffers, this returns all frames for the channel.
    #[inline]
    pub fn channel(&self, band: usize, channel: usize) -> &[T] {
        debug_assert!(band < self.num_bands);
        debug_assert!(channel < self.num_allocated_channels);
        let start = self.offset(channel, band);
        &self.data[start..start + self.num_frames_per_band]
    }

    /// Get a mutable slice for a specific channel and band.
    #[inline]
    pub fn channel_mut(&mut self, band: usize, channel: usize) -> &mut [T] {
        debug_assert!(band < self.num_bands);
        debug_assert!(channel < self.num_allocated_channels);
        let start = self.offset(channel, band);
        &mut self.data[start..start + self.num_frames_per_band]
    }

    /// Get a slice of all bands for a specific channel (contiguous in memory).
    ///
    /// Returns `num_frames` elements (all bands concatenated).
    #[inline]
    pub fn bands(&self, channel: usize) -> &[T] {
        debug_assert!(channel < self.num_allocated_channels);
        let start = channel * self.num_frames;
        &self.data[start..start + self.num_frames]
    }

    /// Get a mutable slice of all bands for a specific channel.
    #[inline]
    pub fn bands_mut(&mut self, channel: usize) -> &mut [T] {
        debug_assert!(channel < self.num_allocated_channels);
        let start = channel * self.num_frames;
        &mut self.data[start..start + self.num_frames]
    }

    /// Get a specific band slice within a channel's band data.
    #[inline]
    pub fn band(&self, channel: usize, band: usize) -> &[T] {
        // Same data as channel(band, channel), just different argument order
        self.channel(band, channel)
    }

    /// Get a mutable specific band slice within a channel's band data.
    #[inline]
    pub fn band_mut(&mut self, channel: usize, band: usize) -> &mut [T] {
        self.channel_mut(band, channel)
    }

    /// Raw access to the underlying data.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable raw access to the underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_band_basic() {
        let buf = ChannelBuffer::<f32>::new_single_band(480, 2);
        assert_eq!(buf.num_frames(), 480);
        assert_eq!(buf.num_frames_per_band(), 480);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.num_bands(), 1);
        assert_eq!(buf.size(), 960);
    }

    #[test]
    fn multi_band_dimensions() {
        let buf = ChannelBuffer::<f32>::new(480, 2, 3);
        assert_eq!(buf.num_frames(), 480);
        assert_eq!(buf.num_frames_per_band(), 160);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.num_bands(), 3);
        assert_eq!(buf.size(), 960);
    }

    #[test]
    fn channel_access_single_band() {
        let mut buf = ChannelBuffer::<f32>::new_single_band(4, 2);
        // Write to channel 0
        let ch0 = buf.channel_mut(0, 0);
        ch0[0] = 1.0;
        ch0[1] = 2.0;
        ch0[2] = 3.0;
        ch0[3] = 4.0;
        // Write to channel 1
        let ch1 = buf.channel_mut(0, 1);
        ch1[0] = 10.0;
        ch1[1] = 20.0;

        assert_eq!(buf.channel(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buf.channel(0, 1), &[10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn multi_band_layout() {
        // 2 channels, 2 bands, 4 frames total (2 per band)
        // Layout: [b0ch0(2) | b1ch0(2) | b0ch1(2) | b1ch1(2)]
        let mut buf = ChannelBuffer::<i16>::new(4, 2, 2);

        // Band 0, channel 0
        buf.channel_mut(0, 0).copy_from_slice(&[1, 2]);
        // Band 1, channel 0
        buf.channel_mut(1, 0).copy_from_slice(&[3, 4]);
        // Band 0, channel 1
        buf.channel_mut(0, 1).copy_from_slice(&[5, 6]);
        // Band 1, channel 1
        buf.channel_mut(1, 1).copy_from_slice(&[7, 8]);

        // Verify the contiguous layout matches C++:
        // [b0ch0, b1ch0, b0ch1, b1ch1]
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5, 6, 7, 8]);

        // bands(channel) returns all bands for a channel concatenated
        assert_eq!(buf.bands(0), &[1, 2, 3, 4]);
        assert_eq!(buf.bands(1), &[5, 6, 7, 8]);

        // band(channel, band) returns a specific band slice
        assert_eq!(buf.band(0, 0), &[1, 2]);
        assert_eq!(buf.band(0, 1), &[3, 4]);
        assert_eq!(buf.band(1, 0), &[5, 6]);
        assert_eq!(buf.band(1, 1), &[7, 8]);
    }

    #[test]
    fn set_num_channels() {
        let mut buf = ChannelBuffer::<f32>::new_single_band(10, 4);
        assert_eq!(buf.num_channels(), 4);
        buf.set_num_channels(2);
        assert_eq!(buf.num_channels(), 2);
        // Data for all 4 channels is still allocated
        assert_eq!(buf.size(), 40);
    }

    #[test]
    #[should_panic(expected = "exceeds allocated")]
    fn set_num_channels_too_large_panics() {
        let mut buf = ChannelBuffer::<f32>::new_single_band(10, 2);
        buf.set_num_channels(3);
    }

    #[test]
    fn zero_initialized() {
        let buf = ChannelBuffer::<f32>::new(480, 2, 3);
        for &v in buf.data() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "divisible by num_bands")]
    fn non_divisible_frames_panics() {
        let _ = ChannelBuffer::<f32>::new(481, 1, 3);
    }

    #[test]
    fn three_band_48khz() {
        // Typical 48kHz configuration: 480 frames, 3 bands of 160
        let buf = ChannelBuffer::<f32>::new(480, 1, 3);
        assert_eq!(buf.num_frames_per_band(), 160);
        assert_eq!(buf.channel(0, 0).len(), 160);
        assert_eq!(buf.channel(1, 0).len(), 160);
        assert_eq!(buf.channel(2, 0).len(), 160);
        assert_eq!(buf.bands(0).len(), 480);
    }
}
