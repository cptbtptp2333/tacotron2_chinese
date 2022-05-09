#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2021-12-10 17:45:54
# @LastEditors  : 任洁
# @LastEditTime : 2022-05-01 16:46:50
# @FilePath     : /Desktop/GST-Tacotron/text/__init__.py

""" from https://github.com/keithito/tacotron """
import re
import unicodedata
from text.symbols import symbols
import text.cleaners

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                  if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    # text = re.sub("[^{}]".format(symbols), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def text_to_sequence_mandarin(text, cleaner_names):
    # print(text)
    # text = text_normalize(text)
    # print(text)
    text = _clean_text(text, cleaner_names)
    # print(text)
    result = [_symbol_to_id[ch] for ch in text]
    # print(result)
    return result


def sequence_to_text_mandarin(sequence):
    '''Converts a sequence of IDs back to a string'''
    """
    with open('mandarin/vocab.json', 'r') as file:
            data = json.load(file)
    idx2char = data['idx2char']
    """
    result = [_id_to_symbol[idx] for idx in sequence]
    return result


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


if __name__ == '__main__':
    import pinyin
    # from config import text_cleaners
    text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
    text = pinyin.get(text, format="numerical", delimiter=" ")
    print(text)
    text_cleaners = ['basic_cleaners']
    sequence = text_to_sequence_mandarin(text, text_cleaners)
    print(sequence)