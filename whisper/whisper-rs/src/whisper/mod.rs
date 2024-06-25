mod audio;
mod tokenizers;
mod utils;

use audio::{get_mel_filteres, read_audio};
use ndarray::{
    concatenate, s, Array, Array2, Array3, ArrayView, ArrayView2, ArrayView3, Axis, Dim, Dimension,
    Ix, StrideShape,
};
use rten::{Input, Model, NodeId, Output};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::fmt;
use tokenizers::Tokenizer;
use utils::{KVCache, Options};

/// Convert an ndarray view into an RTen NdTensorView.
fn as_ndtensor_view<'a, T, const N: usize>(
    view: ArrayView<'a, T, Dim<[Ix; N]>>,
) -> Option<NdTensorView<'a, T, N>>
where
    Dim<[Ix; N]>: Dimension,
{
    view.to_slice().map(|slice| {
        let shape: [usize; N] = view.shape().try_into().unwrap();
        NdTensorView::from_data(shape, slice)
    })
}

/// Convert an owned RTen NdTensor into an ndarray array.
fn into_array<T, const N: usize>(tensor: NdTensor<T, N>) -> Array<T, Dim<[Ix; N]>>
where
    T: Clone,
    Dim<[Ix; N]>: Dimension,
    [usize; N]: Into<StrideShape<Dim<[Ix; N]>>>,
{
    let shape = tensor.shape();
    let data = tensor.into_data();
    Array::from_shape_vec(shape, data).unwrap()
}

pub struct Whisper {
    encoder: Model,
    decoder: Model,
    tokenizer: Tokenizer,
    mel_filters: Array2<f32>,
    options: Options,
}

impl fmt::Debug for Whisper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Whisper").finish()
    }
}

trait Recognition {
    fn get_encoder(&self) -> &Model;

    fn get_decoder(&self) -> &Model;

    fn get_tokenizer(&self) -> &Tokenizer;

    fn get_mel_filters(&self) -> &Array2<f32>;

    fn get_options(&self) -> &Options;

    fn get_default_kvcache(&self) -> KVCache;

    fn inference_logits(
        &self,
        tokens: ArrayView2<i32>,
        decoder_inputs: &[(NodeId, Output)],
        kv_cache: &KVCache,
        initial_token_length: usize,
    ) -> (Array3<f32>, KVCache);

    fn get_audio_features(&self, segments: Vec<Array2<f32>>) -> Array3<f32> {
        let mels: Array3<f32> = segments
            .into_iter()
            .fold(None, |acc, array| {
                Some(match acc {
                    Some(concatenated) => {
                        concatenate![Axis(0), concatenated, array.insert_axis(Axis(0))]
                    }
                    None => array.insert_axis(Axis(0)),
                })
            })
            .unwrap();
        let inputs = as_ndtensor_view(mels.view()).unwrap();
        let encoder_out = self.get_encoder().run_one(inputs.into(), None).unwrap();
        let result: NdTensor<f32, 3> = encoder_out.try_into().unwrap();
        into_array(result)
    }

    fn get_initial_tokens(&self, prompt: Vec<i32>, language: &str) -> Vec<i32> {
        let lang_token = *self.get_tokenizer().lang2token.get(language).unwrap();
        let init_tokens: Vec<i32> = vec![50258, lang_token as i32, 50359];

        if prompt.len() > 0 {
            let prev_prompt_len = self.get_options().n_ctx / 2 - 1;
            let prompt_tokens: Vec<i32>;

            if prompt.len() > prev_prompt_len {
                prompt_tokens = prompt[prompt.len() - prev_prompt_len..].to_vec();
            } else {
                prompt_tokens = prompt;
            }

            let tokens: Vec<i32> = vec![self.get_options().sot_prev as i32]
                .into_iter()
                .chain(prompt_tokens.into_iter())
                .collect();
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        } else {
            let tokens = vec![self.get_options().sot_prev as i32];
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        }
    }

    fn inference(
        &self,
        audio_features: ArrayView3<f32>,
        prompt: Vec<i32>,
        language: &str,
    ) -> Vec<i32> {
        let initial_tokens = self.get_initial_tokens(prompt, language);
        let initial_token_length = initial_tokens.len();

        let mut tokens: Array2<i32> = Array::from_vec(initial_tokens).insert_axis(Axis(0));
        let mut kv_cache = self.get_default_kvcache();

        // Precompute parts of the decoder graph that only depend on encoder
        // outputs. This makes each decoder step much faster.
        let decoder = self.get_decoder();
        let logits_id = decoder.node_id("logits").unwrap();
        let audio_features_id = decoder.node_id("audio_features").unwrap();
        let audio_features_tensor = as_ndtensor_view(audio_features.view()).unwrap();
        let decoder_inputs = decoder
            .partial_run(
                &[(audio_features_id, audio_features_tensor.into())],
                &[logits_id],
                None,
            )
            .expect("decoder run failed");

        for _ in 0..224 {
            let logits: Array3<f32>;
            (logits, kv_cache) = self.inference_logits(
                tokens.view(),
                &decoder_inputs,
                &kv_cache,
                initial_token_length,
            );
            let next_word = logits
                .slice(s![.., -1, ..])
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as usize)
                .unwrap();

            if next_word == self.get_options().eot_token
                || tokens.shape()[1] > self.get_options().n_ctx
            {
                break;
            }

            let next_word_array = Array::from_elem((1, 1), next_word as i32);
            tokens = concatenate!(Axis(1), tokens, next_word_array);
        }
        tokens = tokens.slice(s![.., initial_token_length..]).to_owned();
        tokens.into_raw_vec()
    }

    fn encode(&self, mel: Array2<f32>) -> Array3<f32> {
        let num_frames = mel.shape()[1];
        let mut seek = 0;
        let mut segments = vec![];

        while seek < num_frames {
            let segment: Array2<f32>;

            if seek + audio::N_FRAMES < mel.shape()[1] {
                segment = mel.slice(s![.., seek..seek + audio::N_FRAMES]).to_owned();
            } else {
                segment = mel.slice(s![.., seek..]).to_owned();
            }

            segments.push(audio::pad_or_trim(segment, audio::N_FRAMES));
            seek += audio::N_FRAMES;
        }

        self.get_audio_features(segments)
    }

    fn decode_tokens(&self, result: Vec<i32>) -> String {
        self.get_tokenizer().decode(
            result
                .iter()
                .map(|v| *v as usize)
                .filter(|item| item < &50257)
                .collect(),
        )
    }

    fn get_mel(&self, audio_data: Vec<f32>) -> Array2<f32> {
        audio::log_mel_spectrogram(audio_data, self.get_mel_filters().clone())
    }

    fn run(&self, mel: Array2<f32>, language: &str) -> String {
        let audio_features = self.encode(mel);

        let mut result: Vec<i32> = vec![];
        for audio_feature in audio_features.axis_iter(Axis(0)) {
            let audio_feature = audio_feature.insert_axis(Axis(0));
            let tokens = self.inference(audio_feature, result.clone(), language);
            result.extend(tokens.clone());
        }

        self.decode_tokens(result)
    }
}

impl Recognition for Whisper {
    fn get_encoder(&self) -> &Model {
        &self.encoder
    }

    fn get_decoder(&self) -> &Model {
        &self.decoder
    }

    fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn get_mel_filters(&self) -> &Array2<f32> {
        &self.mel_filters
    }

    fn get_options(&self) -> &Options {
        &self.options
    }

    fn get_default_kvcache(&self) -> KVCache {
        KVCache::default(512)
    }

    fn inference_logits(
        &self,
        tokens: ArrayView2<i32>,
        decoder_inputs: &[(NodeId, Output)],
        kv_cache: &KVCache,
        initial_token_length: usize,
    ) -> (Array3<f32>, KVCache) {
        let tokens = if tokens.shape()[1] > initial_token_length {
            tokens.slice_move(s![.., -1]).insert_axis(Axis(0))
        } else {
            tokens
        };

        let tokens_id = self.decoder.node_id("tokens").unwrap();
        let tokens = as_ndtensor_view(tokens.view()).unwrap();

        // Add the inputs which change on each decoder iteration.
        let mut inputs: Vec<(NodeId, Input)> = vec![(tokens_id, tokens.into())];

        // Add the inputs of kv_cache
        inputs.extend((0..kv_cache.value.len()).map(|idx| {
            if idx % 2 == 0 {
                (
                    self.decoder
                        .node_id(format!("k{}", idx / 2).as_str())
                        .unwrap(),
                    as_ndtensor_view(kv_cache.value[idx].view()).unwrap().into(),
                )
            } else {
                (
                    self.decoder
                        .node_id(format!("v{}", idx / 2).as_str())
                        .unwrap(),
                    as_ndtensor_view(kv_cache.value[idx].view()).unwrap().into(),
                )
            }
        }));

        // Add the inputs which are constant while decoding a chunk of audio.
        inputs.extend(
            decoder_inputs
                .iter()
                .map(|(node_id, output)| (*node_id, output.into())),
        );

        let logits_id = self.decoder.node_id("logits").unwrap();

        let mut outputs: Vec<NodeId> = vec![logits_id];
        outputs.extend((0..kv_cache.value.len()).map(|idx| {
            if idx % 2 == 0 {
                self.decoder
                    .node_id(format!("output_k{}", idx / 2).as_str())
                    .unwrap()
            } else {
                self.decoder
                    .node_id(format!("output_v{}", idx / 2).as_str())
                    .unwrap()
            }
        }));
        let result: [rten::Output; 13] = self
            .decoder
            .run_n(&inputs, outputs.try_into().unwrap(), None)
            .unwrap();

        let (logits, kv_cache) = result.split_at(1);

        let logits: NdTensor<f32, 3> = logits[0].clone().try_into().unwrap();
        let logits = into_array(logits);

        let new_kv_cache = KVCache {
            value: kv_cache
                .to_vec()
                .into_iter()
                .map(|element| {
                    let element: NdTensor<f32, 3> = element.try_into().unwrap();
                    into_array(element)
                })
                .collect(),
        };

        (logits, new_kv_cache)
    }
}

impl Whisper {
    pub fn new(
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
        mel_filters_path: &str,
    ) -> Whisper {
        let encoder = Model::load_file(encoder_path).unwrap();
        let decoder = Model::load_file(decoder_path).unwrap();
        let tokenizer = Tokenizer::new(tokenizer_path);
        let mel_filters = get_mel_filteres(mel_filters_path);
        let options = Options::new(6);

        Whisper {
            encoder,
            decoder,
            tokenizer,
            mel_filters,
            options,
        }
    }

    pub fn recognize_from_audio(&self, audio_path: &str, language: &str) -> String {
        let audio_data = read_audio(audio_path).unwrap();
        let mel = self.get_mel(audio_data);
        self.run(mel, language)
    }
}
