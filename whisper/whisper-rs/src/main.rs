mod whisper;
use std::time::Instant;

fn main() {
    let whisper = whisper::Whisper::new(
        "../weights/base_encoder.rten",
        "../weights/base_decoder.rten",
        "../weights/multilingual.tiktoken",
        "../weights/mel_filters.npz",
    );
    let start = Instant::now();
    let result = whisper.recognize_from_audio("../data/audio.wav", "en");

    let duration = start.elapsed();

    println!("{}", result);

    println!("{:?}", duration);
}
