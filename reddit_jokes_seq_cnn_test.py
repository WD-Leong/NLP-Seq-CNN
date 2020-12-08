
import time
import numpy as np
import pickle as pkl

import tensorflow as tf
import tf_ver2_seq_cnn as tf_model

# Model Parameters. #
seq_length = 30
kernel_sz  = 3
num_stacks = 2
num_layers = 4
hidden_size = 256

model_ckpt_dir = \
    "C:/Users/admin/Desktop/TF_Models/seq_cnn_reddit"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
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
print("Building the Seq CNN Model.")
start_time = time.time()

gpt_model = tf_model.build_model(
    hidden_size, vocab_size, num_layers, 
    n_stacks=num_stacks, kernel_size=kernel_sz)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)
print(gpt_model.summary())

elapsed_time = (time.time()-start_time) / 60
print("Seq CNN Model Built", "("+str(elapsed_time)+" mins).")

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
print("Testing the Seq CNN Model", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
tmp_phrase = ""
while True:
    tmp_phrase = input("Enter a phrase: ")
    if tmp_phrase == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token
        tmp_p_tokens = tmp_phrase.lower().split(" ")
        
        tmp_p_idx = [word2idx.get(
            x, UNK_token) for x in tmp_p_tokens]
        n_tokens  = len(tmp_p_idx)
        tmp_test_in[0, :n_tokens] = tmp_p_idx
        
        tmp_infer = tf_model.infer(
            gpt_model, tmp_test_in[:, :n_tokens], seq_length)
        gen_phrase = [idx2word[x] for x in tmp_infer[0].numpy()]
        gen_phrase = " ".join(gen_phrase)
        
        print("")
        print("Input Phrase:")
        print(" ".join(tmp_p_tokens[:n_tokens]))
        print("Generated Phrase:")
        print(gen_phrase)
        print("-" * 50)
        del n_tokens

