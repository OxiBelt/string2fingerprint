//! Trains the model of the fixture while the fixture is compiled.

use string2fingerprint::build::{ModelBuild, Source};
use string2fingerprint::eval::Objective;
use string2fingerprint::train::TrainConfig;
use string2fingerprint::version::VersionReq;
use string2fingerprint::{Threshold, Thresholds, VERSION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small network that a few dozen steps fit to the development set.
    let mut config = TrainConfig::default();
    config.hidden = vec![16];
    config.epochs = 40;
    config.quantized_epochs = 5;
    config.batch_size = 16;
    config.learning_rate = 1e-2;
    config.input_dropout = 0.0;

    let built = ModelBuild::new("fixture")
        .train_on(Source::benign_lines("../data/benign.txt"))
        .train_on(Source::labeled_lines("../data/attacks.tsv"))
        .config(config)
        .hold_out(0.2)
        .calibrate(
            Objective::MinCost {
                false_positive: 1.0,
                false_negative: 1.0,
            },
            Thresholds::uniform(Threshold::at(128)),
        )
        // The model states the versions of the library that may load it.
        .library_requirement(format!(">={VERSION}").parse::<VersionReq>()?)
        .run()?;

    // A build script can fail the build when the model is not good enough.
    let evaluation = built.evaluation().ok_or("no sample was held out")?;
    if evaluation.samples() == 0 || built.training_samples() < 100 {
        return Err("the development set was not read".into());
    }
    Ok(())
}
