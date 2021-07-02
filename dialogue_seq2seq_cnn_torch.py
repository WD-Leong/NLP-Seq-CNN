
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

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = True
display_step  = 50
cooling_step  = 1000
warmup_steps  = 1000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/TF_Models/dialogue_seq2seq_cnn_torch"
train_loss_file = "C:/Users/admin/Desktop/Codes/"
train_loss_file += "train_loss_dialogue_seq2seq_cnn_torch.csv"

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
if restore_flag:
    ckpt = torch.load(model_ckpt_dir)
    n_iter = ckpt["step"]
    
    seq2seq_model.load_state_dict(ckpt['model_state_dict'])
    seq2seq_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    n_iter = 0
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros([batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros([batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

print("-" * 50)
print("Training the Seq2Seq CNN Network", 
      "(" + str(n_iter), "iterations).")
print("Vocabulary Size:", str(vocab_size))
print("No. of data:", str(len(data_tuple)))
print("-" * 50)

tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_tok = data_tuple[tmp_index][0].split(" ")
        tmp_o_tok = data_tuple[tmp_index][1].split(" ")
        
        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        tmp_o_idx = [word2idx.get(x, UNK_token) for x in tmp_o_tok]
        
        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        n_decode = n_output + 1
        
        tmp_input[n_index, :n_input] = tmp_i_idx
        tmp_seq_out[n_index, 1:n_decode] = tmp_o_idx
        tmp_seq_out[n_index, n_decode] = EOS_token
    
    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = seq2seq_cnn.train_step(
        seq2seq_model, sub_batch, 
        tmp_input, tmp_decode, tmp_output, 
        seq2seq_optimizer, learning_rate=learn_rate_val)
    
    n_iter += 1
    tot_loss += tmp_loss
    if n_iter % display_step == 0:
        end_time = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_time - start_tm) / 60
        start_tm   = time.time()
        
        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data, size=1)
        tmp_data  = data_tuple[sample_id[0]]
        
        tmp_i_tok = tmp_data[0].split(" ")
        tmp_o_tok = tmp_data[1].split(" ")
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
        
        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(learn_rate_val)+".")
        
        print("")
        print("Input Phrase:")
        print(tmp_data[0])
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_data[1])
        
        # Save the training progress. #
        train_loss_list.append((
            n_iter, avg_loss, 
            tmp_data[0], gen_phrase, tmp_data[1]))
        print("-" * 50)
    
    if n_iter % cooling_step == 0:
        print("Cooling the CPU for 2 minutes.")
        
        # Save the model. #
        train_cols_df = [
            "n_iter", "xent_loss", 
            "input_phrase", "gen_phrase", "out_phrase"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=train_cols_df)
        train_loss_df.to_csv(train_loss_file, index=False)
        
        torch.save({
            'step': n_iter,
            'model_state_dict': seq2seq_model.state_dict(),
            'optimizer_state_dict': seq2seq_optimizer.state_dict()
            }, model_ckpt_dir)
        print("Saved model to:", model_ckpt_dir)
        
        time.sleep(120)
        print("Resuming training.")


