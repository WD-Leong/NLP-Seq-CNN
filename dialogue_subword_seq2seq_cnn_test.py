
import time
import numpy as np
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tf_ver2_seq2seq_cnn as tf_seq2seq_cnn

# Model Parameters. #
seq_encode = 50
seq_decode = 51
kernel_size = 5
num_layers  = 4
num_stacks  = 2
prob_keep   = 0.9
hidden_size = 256

tmp_path = "C:/Users/admin/Desktop/"
model_ckpt_dir = tmp_path +\
    "TF_Models/dialogue_subword_seq2seq_cnn"

tmp_pkl_file = tmp_path + "Codes/movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2subword = pkl.load(tmp_file_load)
    subword2idx = pkl.load(tmp_file_load)

vocab_size = len(subword2idx)
subword_vocab = [x for x, y in list(subword2idx.items())]
print("Vocabulary Size:", str(vocab_size))
del data_tuple

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
print("Seq2Seq CNN Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optimizer=seq2seq_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
          manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
    
# Placeholders to store the batch data. #
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Testing the Seq2Seq CNN Model", 
      "(" + str(n_iter), "iterations).")

while True:
    tmp_phrase = input("Enter input phrase: ")
    
    if tmp_phrase == "":
        break
    else:
        tmp_phrase = tmp_phrase.lower()
        tmp_test_in[:, :] = PAD_token
        
        tmp_i_idx = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword2idx)
        tmp_in_phrase = " ".join(bpe.bp_decode(
            tmp_i_idx, idx2subword)).replace("<", "").replace(">", "")
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        gen_ids = tf_seq2seq_cnn.infer(
            seq2seq_model, tmp_test_in, SOS_token, seq_decode)
        gen_phrase = bpe.bp_decode(gen_ids.numpy()[0], idx2subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("")


