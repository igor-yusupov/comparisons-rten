use std::error::Error;

use image;
use rten::{Model, Operators};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

enum ModelType {
    ResNet18,
    MobileNet,
    EfficientNet,
}

struct Args {
    model_type: ModelType,
    is_quant: bool,
}

impl Args {
    fn new(model_type: String, is_quant: bool) -> Args {
        let model_type = model_type.as_str();
        let model: ModelType;

        match model_type {
            "resnet18" => model = ModelType::ResNet18,
            "mobilenet" => model = ModelType::MobileNet,
            "efficientnet" => model = ModelType::EfficientNet,
            _ => model = ModelType::ResNet18,
        }

        Args {
            model_type: model,
            is_quant,
        }
    }

    fn get_model_path(self) -> String {
        match self.model_type {
            ModelType::ResNet18 => {
                if self.is_quant {
                    "../weights/resnet18_quant.rten".into()
                } else {
                    "../weights/resnet18_quant.rten".into()
                }
            }
            ModelType::MobileNet => {
                if self.is_quant {
                    "../weights/mobilenet_quant.rten".into()
                } else {
                    "../weights/mobilenet.rten".into()
                }
            }
            ModelType::EfficientNet => {
                if self.is_quant {
                    "../weights/efficientnet_quant.rten".into()
                } else {
                    "../weights/efficientnet.rten".into()
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ChannelOrder {
    Rgb,
    // Bgr,
}

#[derive(Clone, Copy, PartialEq)]
enum DimOrder {
    /// Use "channels-first" order
    Nchw,
    /// Use "channels-last" order
    Nhwc,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut model_type: String = String::from("");
    let mut is_quant: bool = false;
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => {
                model_type = val.string()?;
            }
            Long("quant") => is_quant = true,
            Short('q') => is_quant = true,
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args::new(model_type, is_quant))
}

/// Read an image from `path` into an NCHW or NHWC tensor, depending on
/// `out_dim_order`.
fn read_image<N: Fn(usize, f32) -> f32>(
    path: &str,
    normalize_pixel: N,
    out_chan_order: ChannelOrder,
    out_dim_order: DimOrder,
    out_height: u32,
    out_width: u32,
) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    // Resize the image using the `imageops::resize` function from the `image`
    // crate rather than using RTen's `resize` operator because
    // `imageops::resize` supports antialiasing. This significantly improves
    // output image quality and thus prediction accuracy when the output is
    // small (eg. 224 or 256px).
    //
    // The outputs of `imageops::resize` still don't match PyTorch exactly
    // though, which can lead to small differences in prediction outputs.
    let input_img = image::imageops::resize(
        &input_img,
        out_width,
        out_height,
        image::imageops::FilterType::Triangle,
    );

    let (width, height) = input_img.dimensions();

    // Map input channel index, in RGB order, to output channel index
    let out_chans = match out_chan_order {
        ChannelOrder::Rgb => [0, 1, 2],
        // ChannelOrder::Bgr => [2, 1, 0],
    };

    let mut img_tensor = Tensor::zeros(&[1, 3, height as usize, width as usize]);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel_value = input_img.get_pixel(x, y)[c] as f32 / 255.0;
                let in_val = normalize_pixel(c, pixel_value);
                img_tensor[[0, out_chans[c], y as usize, x as usize]] = in_val;
            }
        }
    }

    if out_dim_order == DimOrder::Nhwc {
        // NCHW => NHWC
        img_tensor.permute(&[0, 3, 2, 1]);
    }

    Ok(img_tensor)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.get_model_path())?;

    let normalize_pixel = |chan, value| {
        let imagenet_mean = &[0.485, 0.456, 0.406];
        let imagenet_std_dev = &[0.229, 0.224, 0.225];
        (value - imagenet_mean[chan]) / imagenet_std_dev[chan]
    };
    let img_tensor = read_image(
        "../data/image.jpeg",
        normalize_pixel,
        ChannelOrder::Rgb,
        DimOrder::Nchw,
        224,
        224,
    )?;
    let logits: NdTensor<f32, 2> = model.run_one(img_tensor.view().into(), None)?.try_into()?;
    match logits.arg_max(1, true)?.data() {
        Some(value) => println!("Got label: {:?}", value),
        None => println!("Didn't find label"),
    }
    Ok(())
}
