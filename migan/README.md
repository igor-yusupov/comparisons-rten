## Quick start

Create "data" directory and put [image](https://www.dropbox.com/scl/fi/zo69vt41zn13gjh0d8qjh/img.png?rlkey=sk87zel1w5bif59385q3mh7aq&st=wx1axyza&dl=1) and [mask](https://www.dropbox.com/scl/fi/mpljpzw6cpcknpe5rprsn/mask.png?rlkey=hrzd85ccqxftoystimcwp88ts&st=nnfwls61&dl=1) there.
Create "weigths" directory and put [onnx](https://www.dropbox.com/scl/fi/iwsmtncn794b4j22xc2xm/migan.onnx?rlkey=b1kpy9ikuod5lavxc8gcdnfd5&st=d44n3413&dl=1) and [rten](https://www.dropbox.com/scl/fi/r25k66uh2u0a65ku4n9p7/migan.rten?rlkey=6ph0kqbe21tkq8cj3vbtadsl0&st=ouxy2p84&dl=1) weights.

### Run rust code:

```
cd migan-rs

cargo run --release
```

### Run python code:

```
cd migan-py

python3 main.py
```

In both cases you will see result in `data` directory
