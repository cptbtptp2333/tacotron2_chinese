# Tacotron 2

A PyTorch implementation of [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017).

![image](https://github.com/foamliu/GST-Tacotron/raw/master/images/model.png)
## 我的修改
1. mandarin：完成了对BZNSYP数据集的预处理。其中：pre_process.py用以生成整个数据集的.csv文件（存储音频路径和对应的拼音文本），并实现训练集和验证集的随机划分。mandarin中的其他文件为生成文件。（test是取的验证集中的几个，debug测试时用的，可忽略）。
2. text：__init__.py中，新写了text_to_sequence_mandarin()和sequence_to_text_mandarin()。numbers.py是参照别人的版本进行了替换，但目前没有调用。cleaner.py中只关注和修改过basic_cleaner()。symbols中，参照别人的版本进行了拼音的编码。
3. model：由于最终想做情感语音合成，模型部分有GST接口，有gst.py文件。但在目前训练baseline中，我把model.py的Tacotron2()中，forward和inference中的gst相关部分都注释掉了。模型的其余部分没有做改动。


## Dataset

BZNSYP. 

## Dependency

- Python 3.6.8
- PyTorch 1.3.0

## Usage
### Data Pre-processing
Extract dataset and generate features:
```bash
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
