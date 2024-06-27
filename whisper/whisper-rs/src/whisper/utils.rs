use ndarray::{Array3, Dim};

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

pub struct KVCache {
    pub value: Vec<Array3<f32>>,
}

impl KVCache {
    pub fn default(n_ctx: usize, kv_len: usize) -> KVCache {
        let shape = Dim([1, 0, n_ctx]);
        let value: Array3<f32> = Array3::zeros(shape);

        KVCache {
            value: vec![value; kv_len],
        }
    }
}
