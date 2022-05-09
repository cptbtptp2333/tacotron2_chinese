#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2021-12-10 17:45:54
# @LastEditors  : 任洁
# @LastEditTime : 2022-05-02 14:47:40
# @FilePath     : /Desktop/GST-Tacotron/demo.py

import matplotlib.pylab as plt
import numpy as np
import pinyin
import soundfile as sf
import torch
import os
import pandas as pd
import config
from mandarin_data_gen import TextMelLoader, TextMelCollate
from models.loss_function import Tacotron2Loss
from utils import text_to_sequence_mandarin, ensure_folder, plot_data, Denoiser, parse_args
from text import sequence_to_text_mandarin

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()
    
    waveglow_path = '/media/cv516/8e2bd071-88a1-41bb-a3dc-b7eadc0c192e/cv516/ceiling_workspace/renjie/waveglow_298000'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
    text = pinyin.get(text, format="numerical", delimiter=" ")
    # print(text)
    sequence = np.array(text_to_sequence_mandarin(text, config.text_cleaners))[None, :]
    # print(sequence) 
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    test_dataset = TextMelLoader(config.csv_dir, 'test', config)
    
    # ref_mel = valid_dataset.get_mel(config.ref_wav)[None, :]

    data_path = os.path.join(config.csv_dir,'BZNSYP_test_data.csv')
    data = pd.read_csv(data_path)
    test_samples = data.values
    
    # sequence  = test_dataset.get_text(test_samples[0][1])[None, :]
    # sequence = torch.autograd.Variable(sequence).cuda().long()
    # print(sequence)
   
    ref_mel = test_dataset.get_mel(test_samples[0][0])[None, :]
    ref_mel = torch.autograd.Variable(np.transpose(ref_mel, (0, 2, 1))).cuda().float()
    # ref_mel = ref_mel.cuda().float()
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence, ref_mel)
    
    """
    # TODO 测试验证集输出
    evaluation = Tacotron2Loss()
    args = parse_args()
    collate_fn = TextMelCollate(config.n_frames_per_step)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn,
                                               pin_memory=False, shuffle=False, num_workers=args.num_workers,
                                               drop_last=True)
    for i, batch in enumerate(test_loader):
        model.zero_grad()
        x, y = model.parse_batch(batch)
        # (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded)
        # 输出文本
        text_padding, input_lengths, mel_padded, max_len, output_lengths = x
        # print(text_padding[0])
        # print(mel_padded.shape)
        text = sequence_to_text_mandarin(text_padding[0].squeeze().cpu().numpy())
        print(text)

        len = len(text_padding[0])
        tmp = sequence[0].cpu().numpy()[:len]
        # tmp = torch.zeros(1, len - 8)
        # sequence[0] = sequence[0].resize_(1, len)
        # print(sequence[0])
        text_padding[0] = torch.from_numpy(tmp)
        text_padding[1] = text_padding[0]

        # x[0] = sequence
        text = sequence_to_text_mandarin(text_padding[0].squeeze().cpu().numpy())
        # print(text_padding)
        # print(text)
        # print(x)
        
        # forward非平行有问题

        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model((text_padding, input_lengths, mel_padded, max_len, output_lengths))  
        # loss = evaluation(model(x), y)
        # print(loss)
        
        # mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence, ref_mel)  # FIXME inference与文本对应不上
        mel_outputs_postnet = mel_outputs_postnet.type(torch.float16)

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio = audio[0].data.cpu().numpy()
        audio = audio.astype(np.float32)

        print('audio.shape: ' + str(audio.shape))
        print(audio)
        sf.write('output.wav', audio, config.sampling_rate, 'PCM_24')
        break
    """

    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))
    
    ensure_folder('images')
    plt.savefig('images/mel_spec.jpg')

    mel_outputs_postnet = mel_outputs_postnet.type(torch.float16)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        # denoiser_strength = 0.0
        # audio = denoiser(audio, denoiser_strength)
    
    
    audio = audio[0].data.cpu().numpy()
    audio = audio.astype(np.float32)

    print('audio.shape: ' + str(audio.shape))
    print(audio)

    sf.write('output.wav', audio, config.sampling_rate, 'PCM_24')
    