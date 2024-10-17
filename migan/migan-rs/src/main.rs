use std::error::Error;

use rten::{InputOrOutput, Model, NodeId};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;

use image;
use image::{ImageBuffer, RgbImage};

#[derive(Clone, Copy, PartialEq)]
enum ChannelOrder {
    Rgb,
    // Bgr,
}

fn read_image(path: &str, out_chan_order: ChannelOrder) -> Result<Tensor<u8>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    let (width, height) = input_img.dimensions();

    // Map input channel index, in RGB order, to output channel index
    let out_chans = match out_chan_order {
        ChannelOrder::Rgb => [0, 1, 2],
        // ChannelOrder::Bgr => [2, 1, 0],
    };

    let mut img_tensor: Tensor<u8> = Tensor::zeros(&[1, 3, height as usize, width as usize]);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel_value = input_img.get_pixel(x, y)[c] as u8;
                img_tensor[[0, out_chans[c], y as usize, x as usize]] = pixel_value;
            }
        }
    }

    Ok(img_tensor)
}

fn read_mask(path: &str) -> Result<Tensor<u8>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_luma8();

    let (width, height) = input_img.dimensions();

    let mut img_tensor: Tensor<u8> = Tensor::zeros(&[1, 1, height as usize, width as usize]);

    for y in 0..height {
        for x in 0..width {
            let pixel_value = 255 - input_img.get_pixel(x, y)[0] as u8;
            img_tensor[[0, 0, y as usize, x as usize]] =
                if pixel_value < 255 { 0 } else { pixel_value };
        }
    }

    Ok(img_tensor)
}

fn main() -> Result<(), Box<dyn Error>> {
    let model = Model::load_file("../weights/migan.rten")?;
    let img_tensor: Tensor<u8> = read_image("../data/img.png", ChannelOrder::Rgb)?;
    let shape = img_tensor.shape();
    let height = shape[2] as u32;
    let width = shape[3] as u32;

    let mask_tensor: Tensor<u8> = read_mask("../data/mask.png")?;

    let img_id = model.node_id("image")?;
    let mask_id = model.node_id("mask")?;

    let result_id = model.node_id("result")?;

    let inputs: Vec<(NodeId, InputOrOutput)> =
        vec![(img_id, img_tensor.into()), (mask_id, mask_tensor.into())];
    let outputs: Vec<NodeId> = vec![result_id];

    let result = model.run(inputs, &outputs, None)?;
    let result_img: Tensor<u8> = result.first().unwrap().clone().into_tensor().unwrap();
    let result_img = result_img.index_axis(0, 0).permuted(&[1, 2, 0]);
    let pixels = result_img.to_vec();

    let img: RgbImage = ImageBuffer::from_vec(width, height, pixels).unwrap();

    img.save("../data/result_rten.png")?;
    Ok(())
}
