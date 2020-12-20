# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:49:10 2019

@author: chiming
"""

import time
import math
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter

import tensorflow as tf

# Custom functions. #
def compute_bleu_score(
    reference_corpus, translated_corpus, max_order=4, smooth=False):
    
    def _get_n_grams(segment, max_order):
        n_gram_counts = Counter()
        for order in range(1, max_order+1):
            for i in range(0, len(segment)-order+1):
                ngram = tuple(segment[i:(i+order)])
                n_gram_counts[ngram] += 1
        return n_gram_counts
    
    matches_by_order = [0]*max_order
    possible_matches_by_order = [0]*max_order
    
    reference_length   = 0
    translation_length = 0
    
    for (references, translation) in \
        zip(reference_corpus, translated_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        
        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_n_grams(reference, max_order)
        translated_ngram_counts = _get_n_grams(translation, max_order)
        
        overlap = translated_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches
    
    precisions = [0]*max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = \
                (matches_by_order[i]+1.0) / possible_matches_by_order[i]
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = \
                    float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0
        
    if min(precisions) > 0:
        p_log_sum = \
            sum((1.0/max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0
    
    tmp_ratio = float(translation_length) / reference_length
    if tmp_ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (1.0/tmp_ratio))
    
    bleu = geo_mean*bp
    return bleu

# Model Parameters. #
batch_size = 256
sub_batch  = 64
attn_flag  = True
seq_length = 21
kernel_sz  = 3
num_stacks = 3
num_layers = 4

prob_keep = 0.9
hidden_size = 256

tmp_path = "C:/Users/admin/Desktop/Codes/"
train_loss_file = "C:/Users/admin/Desktop/Codes/"
if attn_flag:
    import tf_ver2_seq_cnn_attn as tf_model
    model_ckpt_dir  = \
        "C:/Users/admin/Desktop/TF_Models/dialogue_seq_cnn_attn"
    train_loss_file += "train_loss_dialogue_seq_cnn_attn.csv"
else:
    import tf_ver2_seq_cnn as tf_model
    model_ckpt_dir  = \
        "C:/Users/admin/Desktop/TF_Models/dialogue_seq_cnn"
    train_loss_file += "train_loss_dialogue_seq_cnn.csv"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size)+".")

SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the Sequence CNN Model.")
start_time = time.time()

gpt_model = tf_model.build_model(
    hidden_size, vocab_size, 
    num_layers, p_drop=1.0-prob_keep, 
    n_stacks=num_stacks, kernel_size=kernel_sz)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time()-start_time) / 60
print("Sequence CNN Model Built", "("+str(elapsed_time)+" mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Train the Transformer model. #
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Testing the Sequence CNN Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_in_phrase = input("Enter Input Phrase: ")
    if tmp_in_phrase == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token
        
        tmp_in_phrase = tmp_in_phrase.lower()
        tmp_p_tokens  = tmp_in_phrase.split(" ")
        
        tmp_p_idx = [word2idx.get(
            x, UNK_token) for x in tmp_p_tokens]
        n_tokens  = len(tmp_p_idx) + 1
        tmp_test_in[0, :n_tokens] = tmp_p_idx + [SOS_token]
        
        tmp_infer  = tf_model.infer(
            gpt_model, tmp_test_in[:, :n_tokens], seq_length)
        gen_phrase = [idx2word[x] for x in tmp_infer[0].numpy()]
        gen_output = " ".join(gen_phrase[n_tokens:])
        del n_tokens
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Response:")
        print(gen_output)
        print("-" * 50)


