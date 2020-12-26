import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tensorflow as tf

def decode_seq(inp_seq):    
    # Initial states value is coming from the encoder 
    states_val = encoder_model_inf.predict(inp_seq)
    target_seq = np.zeros((1, 1, len(target_chars)))
    target_seq[0, 0, target_char_to_index_dict['\t']] = 1
    translated_sent = ''
    stop_condition = False
    while not stop_condition:
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_target_char = target_index_to_char_dict[max_val_index]
        translated_sent += sampled_target_char
        if ( (sampled_target_char == '\n') or (len(translated_sent) > max_len_target_sent)) :
            stop_condition = True
        target_seq = np.zeros((1, 1, len(target_chars)))
        target_seq[0, 0, max_val_index] = 1
        states_val = [decoder_h, decoder_c]
    return translated_sent


def assertFunctionEquals(translated_sent, source_code):
    try:
        if exec(source_code) == exec(translated_sent):
            return True
        else:
            return False
    except:
        return False
