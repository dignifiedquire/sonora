//! Checks that methods keep the signatures they had in sonora-agc2 0.2.0.

use sonora_agc2::adaptive_digital_gain_controller::{AdaptiveDigitalGainController, FrameInfo};
use sonora_agc2::common::SUB_FRAMES_IN_FRAME;
use sonora_agc2::fixed_digital_level_estimator::FixedDigitalLevelEstimator;
use sonora_agc2::gain_applier::GainApplier;
use sonora_agc2::limiter::Limiter;
use sonora_agc2::noise_level_estimator::NoiseLevelEstimator;

// Generic signatures break 0.2.0 callers that take a function pointer, pass
// an empty slice or mix buffer types in one slice literal.
#[test]
fn methods_keep_published_signatures() {
    let _: fn(&mut AdaptiveDigitalGainController, &FrameInfo, &mut [&mut [f32]]) =
        AdaptiveDigitalGainController::process;
    let _: fn(&mut FixedDigitalLevelEstimator, &[&[f32]]) -> [f32; SUB_FRAMES_IN_FRAME as usize] =
        FixedDigitalLevelEstimator::compute_level;
    let _: fn(&mut GainApplier, &mut [&mut [f32]]) = GainApplier::apply_gain;
    let _: fn(&mut Limiter, &mut [&mut [f32]]) = Limiter::process;
    let _: fn(&mut NoiseLevelEstimator, &[&[f32]]) -> f32 = NoiseLevelEstimator::analyze;
}
