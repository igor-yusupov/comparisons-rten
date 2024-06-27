## Quick start

### Export weights
```
python3 export.py --model_type {tiny, base or small}
```

```
cd weights
rten-convert {encoder_name}
rten-convert {decoder_name}
```

### Run rust code:

```
cd whisper-rs

cargo run --release {tiny, base, small}
```

### Run python code:

```
cd whisper-py

python3 main.py --model_type {tiny, base or small}
```
