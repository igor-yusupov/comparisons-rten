use ndarray::{Array4, Dim};

#[derive(Debug)]
pub struct Options {
    pub eot_token: usize,
    pub sot_prev: usize,
    pub n_ctx: usize,
}

impl Options {
    pub fn new() -> Options {
        Options {
            eot_token: 50257,
            sot_prev: 50361,
            n_ctx: 448,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub kv_cache: Array4<f32>,
}

impl KVCache {
    pub fn default(n_ctx: usize) -> KVCache {
        let shape = Dim([12, 1, 451, n_ctx]);
        let value: Array4<f32> = Array4::zeros(shape);

        KVCache {
            kv_cache: value.clone(),
        }
    }
}
