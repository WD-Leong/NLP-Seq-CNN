
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_seq2seq_cnn as tf_seq2seq_cnn

# Define the subword decoder. #
def bp_decode(indices_in, idx2subword):
    sw_idx_list = [idx2subword[x] for x in indices_in]
    words_list  = []
    
    curr_sw = ""
    for n_sw in range(len(sw_idx_list)):
        tmp_sw = sw_idx_list[n_sw]
        if tmp_sw.find("<") != -1 \
            and tmp_sw.find(">") != -1:
            tmp_word = tmp_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") != -1 \
            and tmp_sw.find("<") == -1:
            curr_sw += tmp_sw
            tmp_word = curr_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") == -1 \
            and tmp_sw.find("<") != -1:
            curr_sw += tmp_sw
        else:
            curr_sw += tmp_sw
    return words_list

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_encode = 25
seq_decode = 26

kernel_size = 5
num_layers  = 4
num_stacks  = 2
prob_keep   = 0.9
hidden_size = 256

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 50000
restore_flag  = True
display_step  = 100
cooling_step  = 1000
warmup_steps  = 1000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "C:/Users/admin/Desktop/"
model_ckpt_dir  = tmp_path +\
    "TF_Models/dialogue_subword_seq2seq_cnn"
train_loss_file = tmp_path +\
    "Codes/train_loss_dialogue_subword_seq2seq_cnn.csv"

tmp_pkl_file = tmp_path + "Codes/movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2subword = pkl.load(tmp_file_load)
    subword2idx = pkl.load(tmp_file_load)

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    if len(tmp_data[0]) <= seq_encode \
        and len(tmp_data[1]) <= (seq_decode-1):
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
vocab_size = len(subword2idx)
print("Vocabulary Size:", str(vocab_size))
del filtered_data

num_data  = len(data_tuple)
SOS_token = subword2idx["<SOS>"]
EOS_token = subword2idx["<EOS>"]
PAD_token = subword2idx["<PAD>"]
UNK_token = subword2idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Seq2Seq CNN Model.")
start_time = time.time()

seq2seq_model = tf_seq2seq_cnn.build_model(
    hidden_size, num_layers, seq_encode, 
    vocab_size, vocab_size, n_stacks=num_stacks, 
    kernel_size=kernel_size, p_drop=1.0-prob_keep)
seq2seq_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("Seq2Seq CNN Model built", 
      "(" + str(elapsed_time) + " mins).")

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
        print("Model restored from {}".format(
              manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros([batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros([batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Seq2Seq CNN Model", 
      "(" + str(n_iter), "iterations).")
print("Total of", str(len(data_tuple)), "training data.")

tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    #annealing_val  = anneal_rate ** ((n_iter+1) // anneal_step)
    #learn_rate_val = max(annealing_val * initial_lr, 1.0e-5)

    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = data_tuple[tmp_index][0]
        tmp_o_idx = data_tuple[tmp_index][1]
        
        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        n_decode = n_output + 1
        
        tmp_input[n_index, :n_input] = tmp_i_idx
        tmp_seq_out[n_index, 1:n_decode] = tmp_o_idx
        tmp_seq_out[n_index, n_decode] = EOS_token
    
    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = tf_seq2seq_cnn.train_step(
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
        
        tmp_i_idx = tmp_data[0]
        tmp_i_tok = bp_decode(tmp_i_idx, idx2subword)
        tmp_o_tok = bp_decode(tmp_data[1], idx2subword)
        
        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        tmp_out_phrase = " ".join(
            tmp_o_tok).replace("<", "").replace(">", "")
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        gen_ids = tf_seq2seq_cnn.infer(
            seq2seq_model, tmp_test_in, SOS_token, seq_decode)
        gen_phrase = bp_decode(gen_ids.numpy()[0], idx2subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(seq2seq_optimizer.lr.numpy()))
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_out_phrase)
        print("")
        
        # Save the training progress. #
        train_loss_list.append((
            n_iter, avg_loss, 
            tmp_in_phrase, gen_phrase, tmp_out_phrase))
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
        print("Cooling the GPU for 3 minutes.")
        time.sleep(180)
        print("Resuming training.")


