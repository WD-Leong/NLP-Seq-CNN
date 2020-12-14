
import time
import math
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter

import tensorflow as tf
import tf_ver2_seq_cnn_attn as tf_model

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
batch_size = 64
sub_batch  = 8
kernel_sz  = 5
seq_length = 500
num_stacks = 3
num_layers = 6

gradient_clip = 1.00
maximum_iter  = 100000
restore_flag  = False
save_step     = 250
warmup_steps  = 1000
infer_step    = 50
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
warmup_flag = True
cooling_step = 250

model_ckpt_dir  = "C:/Users/admin/Desktop/" +\
    "TF_Models/transformer_ver2_wiki_plots_subword"
train_loss_file = "C:/Users/admin/Desktop/Codes/" +\
    "train_loss_transformer_wiki_plots_subword.csv"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/wikipedia_plot_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    idx2subword = pkl.load(tmp_load_file)
    subword2idx = pkl.load(tmp_load_file)

vocab_size = len(subword2idx)
SOS_token = subword2idx["<SOS>"]
EOS_token = subword2idx["<EOS>"]
PAD_token = subword2idx["<PAD>"]
UNK_token = subword2idx["<UNK>"]
print("Vocabulary Size:", str(vocab_size)+".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tmp_data = []
for tmp_row in full_data:
    if len(tmp_row) > 1 and \
        len(tmp_row) <= seq_length:
        tmp_data.append(tmp_row + [EOS_token])
num_data  = len(tmp_data)
print("Total of", str(len(full_data)), "rows loaded.")
print("Total of", str(len(tmp_data)), "rows filtered.")

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

gpt_model = tf_model.build_model(
    hidden_size, vocab_size, 
    num_layers, p_drop=1.0-prob_keep, 
    n_stacks=num_stacks, kernel_size=kernel_sz)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", "("+str(elapsed_time)+" mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

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

# Train the Transformer model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    learning_rate = max(
        anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            learning_rate = max(
                anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_idx = tmp_data[tmp_index]
        n_inputs  = len(tmp_p_idx)
        
        tmp_out_seq[n_index, :n_inputs] = tmp_p_idx
        del tmp_p_idx
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = tf_model.train_step(
        gpt_model, sub_batch, tmp_input, tmp_output, 
        gpt_optimizer, learning_rate=learning_rate)

    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip)+".")
        print("Learning Rate:", str(learning_rate)+".")
        print("Average Loss:", str(avg_loss)+".")
        train_loss_list.append((n_iter, avg_loss))
        
        if n_iter % infer_step != 0:
            start_tm = time.time()
            print("-" * 50)
        
        if n_iter % infer_step == 0:
            tmp_test_in[:, :] = PAD_token
            sample_test = np.random.choice(num_data, size=1)
            tmp_p_index = tmp_data[sample_test[0]]
            
            in_phrase = bp_decode(tmp_p_index, idx2subword)
            in_phrase = " ".join(
                in_phrase).replace("<", "").replace(">", "")
            
            n_tokens = len(tmp_p_index)
            n_sample = np.random.randint(1, high=n_tokens-1)
            tmp_test_in[0, :n_sample] = tmp_p_index[:n_sample]
            
            tmp_infer = tf_model.infer(
                gpt_model, tmp_test_in[:, :n_sample], seq_length)
            del sample_test, n_tokens
            
            gen_phrase = bp_decode(tmp_infer[0].numpy(), idx2subword)
            gen_phrase = \
                " ".join(gen_phrase).replace("<", "").replace(">", "")
            
            test_phrase = \
                bp_decode(tmp_p_index[:n_sample], idx2subword)
            test_phrase = \
                " ".join(test_phrase).replace("<", "").replace(">", "")
            del tmp_p_index
            
            print("")
            print("Input Phrase:")
            print(test_phrase)
            print("Generated Phrase:")
            print(gen_phrase)
            print("Actual Phrase:")
            print(in_phrase)
            del n_sample
            print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 5 minutes.")
        time.sleep(300)
        print("Resume Training.")
        print("-" * 50)

