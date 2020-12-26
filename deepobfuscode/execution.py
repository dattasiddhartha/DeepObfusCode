import json 
import numpy as np
from collections import Counter
from util import *
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import load_model
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def obfuscated_execution(obfuscated_code, target_chars, target_index_to_char_dict, target_char_to_index_dict, max_len_target_sent):

    source_sentences = []
    source_chars = set()
    nb_samples = 1
    source_line = str(obfuscated_code).split('\t')[0]
    source_sentences.append(source_line)
    for ch in source_line:
        if (ch not in source_chars):
            source_chars.add(ch)
    source_chars = sorted(list(source_chars))
    source_index_to_char_dict = {}
    source_char_to_index_dict = {}
    for k, v in enumerate(source_chars):
        source_index_to_char_dict[k] = v
        source_char_to_index_dict[v] = k
    source_sent = source_sentences
    max_len_source_sent = max([len(line) for line in source_sent])

    tokenized_source_sentences = np.zeros(shape = (nb_samples,max_len_source_sent,len(source_chars)), dtype='float32')
    for i in range(nb_samples):
        for k,ch in enumerate(source_sent[i]):
            tokenized_source_sentences[i,k,source_char_to_index_dict[ch]] = 1

    model = load_model('decryption_key.h5')
    encoder_model_inf = load_model('encoder_decryption_key.h5')
    decoder_model_inf = load_model('decoder_decryption_key.h5')

    for seq_index in range(1):
        inp_seq = tokenized_source_sentences[seq_index:seq_index+1]
        translated_code = decode_seq(inp_seq)
        print('-')
        print('Input sentence:', source_sent[seq_index])
        print('Decoded sentence:', translated_code)
    exec(translated_code)

