# Tacotron 2

A PyTorch implementation of [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017).

![image](https://github.com/foamliu/GST-Tacotron/raw/master/images/model.png)

## Dataset

BZNSYP. 

## Dependency

- Python 3.6.8
- PyTorch 1.3.0

## Usage
### Data Pre-processing
Extract dataset and generate features:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Generate mel-spectrogram for text "For the first time in her life she had been danced tired."
```bash
$ python demo.py
```
![image](https://github.com/foamliu/Tacotron2-CN/raw/master/images/mel_spec.jpg)
