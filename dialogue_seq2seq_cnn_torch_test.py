
import time
import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch_seq2seq_cnn_module as seq2seq_cnn

# Set benchmark to be true. #
torch.backends.cudnn.benchmark = True

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_encode = 10
seq_decode = 11

kernel_size = 3
num_layers  = 4
num_stacks  = 3
prob_keep   = 0.9
hidden_size = 256

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/TF_Models/dialogue_seq2seq_cnn_torch"

tmp_pkl_file = tmp_path + "movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2word = pkl.load(tmp_file_load)
    word2idx = pkl.load(tmp_file_load)
vocab_size = len(word2idx)

num_data  = len(data_tuple)
SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads to use. #
torch.set_num_threads(1)

print("Building the Sequence CNN Model.")
start_time = time.time()

seq2seq_model = seq2seq_cnn.Seq2Seq_CNN_Network(
    hidden_size, num_layers, vocab_size, vocab_size, 
    n_stacks=num_stacks, p_drop=1.0-prob_keep, 
    kernel_size=kernel_size, attn_type="mult_attn")

if torch.cuda.is_available():
    seq2seq_model.cuda()
seq2seq_optimizer = torch.optim.AdamW(
    seq2seq_model.parameters())

elapsed_time = (time.time() - start_time) / 60
print("Seq2Seq CNN Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = torch.load(model_ckpt_dir)
n_iter = ckpt["step"]

seq2seq_model.load_state_dict(ckpt['model_state_dict'])
seq2seq_optimizer.load_state_dict(ckpt['optimizer_state_dict'])

# Placeholders to store the batch data. #
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

print("-" * 50)
print("Testing the Seq2Seq CNN Network", 
      "(" + str(n_iter), "iterations).")
print("Vocabulary Size:", str(vocab_size))
print("-" * 50)

while True:
    tmp_phrase = input("Enter input phrase: ")
    if tmp_phrase == "":
        break
    else:
        tmp_phrase = tmp_phrase.lower()
        
        tmp_test_in[:, :] = PAD_token
        tmp_i_tok = tmp_phrase.split(" ")
        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        infer_in = torch.tensor(
            tmp_test_in, dtype=torch.long)
        if torch.cuda.is_available():
            infer_in = infer_in.cuda()
        
        gen_ids = seq2seq_model.infer(
            infer_in, SOS_token, seq_decode)
        if torch.cuda.is_available():
            gen_ids = gen_ids.detach().cpu()
        
        gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase = " ".join(gen_phrase[1:])
        
        print("")
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("-" * 50)

