#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2021-12-10 17:45:54
# @LastEditors  : 任洁
# @LastEditTime : 2022-05-01 15:36:45
# @FilePath     : /Desktop/GST-Tacotron/config.py

import torch
import json
from text.symbols import symbols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

wav_folder = '/home/renjie/Desktop/data/train/segmented/wavn'
tran_file = '/home/renjie/Desktop/data/train/segmented/prompts.gui'

csv_dir = "/media/cv516/8e2bd071-88a1-41bb-a3dc-b7eadc0c192e/cv516/ceiling_workspace/renjie/mandarin"

num_train = 9900
num_valid = 100

unk_id = 0

################################
# Experiment Parameters        #
################################
epochs = 500
iters_per_checkpoint = 1000
seed = 1234
dynamic_loss_scaling = True
fp16_run = False
distributed_run = False

################################
# Data Parameters             #
################################
load_mel_from_disk = False

################################
# Audio Parameters             #
################################
max_wav_value = 32768.0
sampling_rate = 48000
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################

symbols_embedding_dim = 512

# vocab_file = '/home/renjie/Desktop/GST-Tacotron/mandarin/vocab.json'
# with open(vocab_file, 'r') as file:
    #    data = json.load(file)
# char2idx = data['char2idx']

n_symbols = len(symbols) 
text_cleaners = ['basic_cleaners']
# Reference encoder
ref_enc_filters = [32, 32, 64, 64, 128, 128]
ref_wav = 'ref_wav/000001.wav'

# Style token layer
token_num = 10
token_emb_size = 512
num_heads = 8

use_neutral = True  # TODO
pretrain_ser_path = '/media/cv516/8e2bd071-88a1-41bb-a3dc-b7eadc0c192e/cv516/ceiling_workspace/renjie/gst_ESD_stl_80-0.7601.pth'
classify_num = 512
classify_filters = [classify_num, 5]
dropout_p = 0.2  # Dropout参数

# Encoder parameters
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_embedding_dim = 512  # TODO default = 256

# Decoder parameters
n_frames_per_step = 1  # currently only 1 is supported
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000  # TODO: default=1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location Layer parameters
attention_location_n_filters = 32
attention_location_kernel_size = 31

# Mel-post processing network parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 1e-3
weight_decay = 1e-6
batch_size = 32
mask_padding = True  # set model's padded outputs to padded values
