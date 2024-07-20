mod whisper;
use std::time::Instant;

use whisper::WhisperVersion;

struct Args {
    model_version: WhisperVersion,
}

impl Args {
    fn new(model_version: String) -> Args {
        let model_version = model_version.as_str();
        let version: WhisperVersion;

        match model_version {
            "tiny" => version = WhisperVersion::Tiny,
            "base" => version = WhisperVersion::Base,
            "small" => version = WhisperVersion::Small,
            "medium" => version = WhisperVersion::Medium,
            _ => version = WhisperVersion::Base,
        }

        Args {
            model_version: version,
        }
    }
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut model_version: String = String::from("");
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => {
                model_version = val.string()?;
            }
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args::new(model_version))
}

fn test_whisper(model_version: WhisperVersion) {
    let whisper_model: whisper::Whisper;
    match model_version {
        WhisperVersion::Tiny => {
            whisper_model = whisper::Whisper::new(
                "../weights/tiny_encoder.rten",
                "../weights/tiny_decoder.rten",
                "../assets/multilingual.tiktoken",
                "../assets/mel_filters.npz",
                model_version,
            );
        }
        WhisperVersion::Base => {
            whisper_model = whisper::Whisper::new(
                "../weights/base_encoder.rten",
                "../weights/base_decoder.rten",
                "../assets/multilingual.tiktoken",
                "../assets/mel_filters.npz",
                model_version,
            );
        }
        WhisperVersion::Small => {
            whisper_model = whisper::Whisper::new(
                "../weights/small_encoder.rten",
                "../weights/small_decoder.rten",
                "../assets/multilingual.tiktoken",
                "../assets/mel_filters.npz",
                model_version,
            );
        }
        WhisperVersion::Medium => {
            whisper_model = whisper::Whisper::new(
                "../weights/medium_encoder.rten",
                "../weights/medium_decoder.rten",
                "../assets/multilingual.tiktoken",
                "../assets/mel_filters.npz",
                model_version,
            );
        }
    }

    let start = Instant::now();
    let result = whisper_model.recognize_from_audio("../data/audio.wav", "en");
    let duration = start.elapsed();

    println!("{}", result);

    println!("{:?}", duration);
}

fn main() {
    let args = parse_args().unwrap();
    test_whisper(args.model_version);
}
