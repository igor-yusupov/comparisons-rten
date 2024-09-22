## Quick start

Download [weights](https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx) and put them to `weights` directory
Create `data` directory and put there some image with name `image.jpeg`

### Export weights

```
cd weights
rten-convert {model_name}
```

### Run rust code:
https://github.com/robertknight/rten/blob/main/rten-examples/src/rmbg.rs


### Run python code:

```
cd rmbg-py

python3 main.py --weights_type {fp32, fp16, int8}
```
