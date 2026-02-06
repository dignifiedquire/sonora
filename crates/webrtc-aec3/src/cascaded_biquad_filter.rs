//! Cascaded biquad (IIR) filter — direct form 1.
//!
//! Ported from `modules/audio_processing/utility/cascaded_biquad_filter.h/cc`.

/// Coefficients for a single second-order section.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BiQuadCoefficients {
    pub b: [f32; 3],
    pub a: [f32; 2],
}

/// State for a single biquad section.
#[derive(Debug, Clone)]
struct BiQuad {
    coefficients: BiQuadCoefficients,
    x: [f32; 2],
    y: [f32; 2],
}

impl BiQuad {
    fn new(coefficients: BiQuadCoefficients) -> Self {
        Self {
            coefficients,
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn reset(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// Cascaded biquad filter applying multiple second-order sections in series.
pub(crate) struct CascadedBiQuadFilter {
    biquads: Vec<BiQuad>,
}

impl CascadedBiQuadFilter {
    pub(crate) fn new(coefficients: &[BiQuadCoefficients]) -> Self {
        Self {
            biquads: coefficients.iter().map(|c| BiQuad::new(*c)).collect(),
        }
    }

    /// Filters `x` into `y` (separate input/output).
    pub(crate) fn process(&mut self, x: &[f32], y: &mut [f32]) {
        if self.biquads.is_empty() {
            y.copy_from_slice(x);
            return;
        }
        Self::apply_biquad(x, y, &mut self.biquads[0]);
        for k in 1..self.biquads.len() {
            // Split borrow: process y in-place through remaining stages.
            let (_, rest) = self.biquads.split_at_mut(k);
            let bq = &mut rest[0];
            // In-place: read from y, write to y.
            let c_b_0 = bq.coefficients.b[0];
            let c_b_1 = bq.coefficients.b[1];
            let c_b_2 = bq.coefficients.b[2];
            let c_a_0 = bq.coefficients.a[0];
            let c_a_1 = bq.coefficients.a[1];
            let mut m_x_0 = bq.x[0];
            let mut m_x_1 = bq.x[1];
            let mut m_y_0 = bq.y[0];
            let mut m_y_1 = bq.y[1];
            for v in y.iter_mut() {
                let tmp = *v;
                *v = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
                m_x_1 = m_x_0;
                m_x_0 = tmp;
                m_y_1 = m_y_0;
                m_y_0 = *v;
            }
            bq.x = [m_x_0, m_x_1];
            bq.y = [m_y_0, m_y_1];
        }
    }

    /// Filters `y` in-place through all stages.
    pub(crate) fn process_in_place(&mut self, y: &mut [f32]) {
        for bq in &mut self.biquads {
            let c_b_0 = bq.coefficients.b[0];
            let c_b_1 = bq.coefficients.b[1];
            let c_b_2 = bq.coefficients.b[2];
            let c_a_0 = bq.coefficients.a[0];
            let c_a_1 = bq.coefficients.a[1];
            let mut m_x_0 = bq.x[0];
            let mut m_x_1 = bq.x[1];
            let mut m_y_0 = bq.y[0];
            let mut m_y_1 = bq.y[1];
            for v in y.iter_mut() {
                let tmp = *v;
                *v = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
                m_x_1 = m_x_0;
                m_x_0 = tmp;
                m_y_1 = m_y_0;
                m_y_0 = *v;
            }
            bq.x = [m_x_0, m_x_1];
            bq.y = [m_y_0, m_y_1];
        }
    }

    pub(crate) fn reset(&mut self) {
        for bq in &mut self.biquads {
            bq.reset();
        }
    }

    fn apply_biquad(x: &[f32], y: &mut [f32], bq: &mut BiQuad) {
        debug_assert_eq!(x.len(), y.len());
        let c_b_0 = bq.coefficients.b[0];
        let c_b_1 = bq.coefficients.b[1];
        let c_b_2 = bq.coefficients.b[2];
        let c_a_0 = bq.coefficients.a[0];
        let c_a_1 = bq.coefficients.a[1];
        let mut m_x_0 = bq.x[0];
        let mut m_x_1 = bq.x[1];
        let mut m_y_0 = bq.y[0];
        let mut m_y_1 = bq.y[1];
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let tmp = *xi;
            *yi = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
            m_x_1 = m_x_0;
            m_x_0 = tmp;
            m_y_1 = m_y_0;
            m_y_0 = *yi;
        }
        bq.x = [m_x_0, m_x_1];
        bq.y = [m_y_0, m_y_1];
    }
}
