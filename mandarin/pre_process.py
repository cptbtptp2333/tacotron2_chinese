#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2022-04-11 17:20:17
# @LastEditors  : 任洁
# @LastEditTime : 2022-05-09 17:31:40
# @FilePath     : /Desktop/GST-Tacotron2/mandarin_pinyin/pre_process.py

import os
import pickle
import random
import sys
sys.path.append('/home/renjie/Desktop/GST-Tacotron')
import pandas as pd
import numpy as np
import json
# import config
# from config import tran_file, wav_folder, num_valid


def generate_text_csv():
    data_dir = "/home/renjie/Desktop/BZNSYP/000001-010000.txt"
    audio_dir = "/home/renjie/Desktop/BZNSYP"
    csv_file = "/home/renjie/Desktop/GST-Tacotron2/mandarin/text_pair.csv"
    dict_list = []
    with open (file=data_dir, mode="r") as f:
        texts = f.readlines()

    char2idx = {}
    idx2char = {}
    
    for i in range(len(texts)):
        if i % 2 == 0:
            index = texts[i].split("\t")[0]
            
            audio_path = os.path.join(audio_dir, "{}.wav".format(index))
        if i % 2 == 1:
            text = texts[i].split("\t").strip("\n")
            
            # 创建vocab
            tokens = text.split()
            for token in tokens:
                if not token in char2idx:
                    next_index = len(char2idx)
                    char2idx[token] = next_index
                    idx2char[next_index] = token

            text_list = {
                        'audio_path': audio_path,
                        'text': text,
                    }
                    
            dict_list.append(text_list)

    data = dict()
    data['char2idx'] = char2idx
    data['idx2char'] = idx2char
    vocab_file = 'vocab.json'

    with open(vocab_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print('vocab_size: ' + str(len(data['char2idx'])))
    
    df = pd.DataFrame(dict_list, columns=["audio_path", "text"])
    df.to_csv(csv_file, index=False)


def random_split_dataset(train2all):
    """
    训练集和验证集类别平均的随机划分
    """
    # 读取文件路径
    csv_dir = "/home/renjie/Desktop/GST-Tacotron2_v2/mandarin"
    csv_path = os.path.join(csv_dir, "text_pair.csv")
    # 输出文件路径
    train_data_path = os.path.join(csv_dir, 'BZNSYP_train_data.csv')
    val_data_path = os.path.join(csv_dir, 'BZNSYP_val_data.csv')

    train_data_dict = []
    val_data_dict = []

    data_info = pd.read_csv(csv_path)
    data = data_info.values
    
    np.random.shuffle(data)

    n = len(data)
    train_num = int(train2all * n)
    
    for i in range(n):
        item = data[i]
        if i < train_num:
            train_one_data_dict = {
                'audio_path': item[0],     
                'text': item[1],
            }
            train_data_dict.append(train_one_data_dict)

        else:
            val_one_data_dict = {
                'audio_path': item[0],     
                'text': item[1],
            }
            val_data_dict.append(val_one_data_dict)

    df_train = pd.DataFrame(train_data_dict, columns=["audio_path", "text"])
    df_train.to_csv(train_data_path, index=False)
    df_val = pd.DataFrame(val_data_dict, columns=["audio_path", "text"])
    df_val.to_csv(val_data_path, index=False)   


def generate_test_csv():
    dict_list = []
    text_list = {
                    'audio_path': '/home/renjie/Desktop/BZNSYP/000001.wav',
                    'text': 'ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1',
                    }
    data_path = os.path.join(config.csv_dir,'BZNSYP_test_data.csv')                
    dict_list.append(text_list)
    df = pd.DataFrame(dict_list, columns=["audio_path", "text"])
    df.to_csv(data_path, index=False)
    

if __name__ == "__main__":
    
    """
    valid = random.sample(samples, num_valid)
    train = []
    for sample in samples:
        if sample not in valid:
            train.append(sample)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open('train.pkl', 'wb') as file:
        pickle.dump(train, file)
    with open('valid.pkl', 'wb') as file:
        pickle.dump(valid, file)
    """
    
    generate_text_csv()
    train2all = 0.99
    random_split_dataset(train2all)
    # generate_test_csv()