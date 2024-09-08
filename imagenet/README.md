## Quick start

Create "data" directory and put any image there with name "image.jpeg".

### Export weights
```
python3 export.py --model_type {resnet18, mobilenet or efficientnet}
```

```
cd weights
rten-convert {resnet18.onnx, mobilenet.onnx or efficientnet.onnx}
```

### Quantization (optional)
```
python3 quant.py --model_type {resnet18, mobilenet or efficientnet}
```

### Run rust code:

```
cd imagenet-rs

cargo run --release {resnet18, mobilenet or efficientnet} --quant (optional)
```

### Run python code:

```
cd imagenet-py

python3 main.py --model_type {resnet18, mobilenet or efficientnet} --quant (optional)
```
