
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_seq2seq_cnn as tf_seq2seq_cnn

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_encode = 10
seq_decode = 11

kernel_size = 5
num_layers  = 4
num_stacks  = 2
prob_keep   = 0.9
hidden_size = 256

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 20000
restore_flag  = True
display_step  = 50
cooling_step  = 1000
warmup_steps  = 1000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/TF_Models/dialogue_seq2seq_cnn"
train_loss_file = "C:/Users/admin/Desktop/Codes/"
train_loss_file += "train_loss_dialogue_seq2seq_cnn.csv"

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
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Sequence CNN Model.")
start_time = time.time()

seq2seq_model = tf_seq2seq_cnn.build_model(
    hidden_size, num_layers, seq_encode, 
    vocab_size, vocab_size, n_stacks=num_stacks, 
    kernel_size=kernel_size, p_drop=1.0-prob_keep)
seq2seq_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print(seq2seq_model.summary())
print("Seq2Seq CNN Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optimizer=seq2seq_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros([batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros([batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Seq2Seq CNN Network", 
      "(" + str(n_iter), "iterations).")
print("Vocabulary Size:", str(vocab_size))
print("No. of data:", str(len(data_tuple)))

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
    
    tmp_loss = tmp_loss = tf_seq2seq_cnn.train_step(
        seq2seq_model, sub_batch, 
        tmp_input, tmp_decode, tmp_output, 
        seq2seq_optimizer, learning_rate=learn_rate_val)
    
    n_iter += 1
    tot_loss += tmp_loss.numpy()
    ckpt.step.assign_add(1)

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
        
        gen_ids = tf_seq2seq_cnn.infer(
            seq2seq_model, tmp_test_in, SOS_token, seq_decode)
        gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase = " ".join(gen_phrase[1:])
        
        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(seq2seq_optimizer.lr.numpy()))
        
        print("")
        print("Input Phrase:")
        print(tmp_data[0])
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_data[1])
        print("")
        
        # Save the training progress. #
        train_loss_list.append((
            n_iter, avg_loss, 
            tmp_data[0], gen_phrase, tmp_data[1]))
        train_cols_df = [
            "n_iter", "xent_loss", 
            "input_phrase", "gen_phrase", "out_phrase"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=train_cols_df)
        train_loss_df.to_csv(train_loss_file, index=False)
        
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        print("-" * 50)
    
    if n_iter % cooling_step == 0:
        print("Cooling the CPU for 3 minutes.")
        time.sleep(180)
        print("Resuming training.")


