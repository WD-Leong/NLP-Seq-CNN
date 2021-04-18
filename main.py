
import time
import numpy as np
import pickle as pkl
from fastapi import FastAPI
import byte_pair_encoding as bpe

import tensorflow as tf
import tf_ver2_seq_cnn_keras as tf_model

# Model Parameters. #
seq_length = 51
kernel_sz  = 3
num_stacks = 3
num_layers = 4
prob_keep  = 0.9
hidden_size = 256

model_ckpt_dir = "C:/Users/admin/Desktop/"
model_ckpt_dir += "TF_Models/dialogue_subword_seq_cnn_keras"

tmp_pkl_file = "C:/Users/admin/Desktop/Codes/"
tmp_pkl_file += "movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    idx2subword = pkl.load(tmp_load_file)
    subword2idx = pkl.load(tmp_load_file)
subword_vocab = [x for x, y in list(subword2idx.items())]
del data_tuple

SOS_token = subword2idx["<SOS>"]
EOS_token = subword2idx["<EOS>"]
PAD_token = subword2idx["<PAD>"]
UNK_token = subword2idx["<UNK>"]
vocab_size = len(subword2idx)
print("Vocabulary Size:", str(vocab_size))

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Seq CNN Model.")
start_time = time.time()

seq_cnn_network = tf_model.Seq_CNN_Network(
    hidden_size, vocab_size, num_layers, 
    p_drop=1.0-prob_keep, n_stacks=num_stacks, 
    kernel_size=kernel_sz, attn_type="mult_attn")
model_optimizer = tf.keras.optimizers.Adam()

# Initialise the model. #
seq_cnn_network.init_model()
elapsed_time = (time.time() - start_time) / 60
print("Seq2Seq CNN Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq_cnn_network=seq_cnn_network, 
    model_optimizer=model_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
          manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Placeholders to store the batch data. #
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Testing the Sequence CNN Model", 
      "(" + str(n_iter), "iterations).")

# API call. #
app = FastAPI()

@app.get("/bot_response/")
async def process_input(phrase: str = ""):
    if phrase == "":
        return {"bot_reply": "No input phrase."}
    else:
        tmp_phrase = phrase.lower().replace("\"", "")
        tmp_test_in[:, :] = PAD_token
        
        tmp_i_idx = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword2idx)
        n_sub_tok = len(tmp_i_idx) + 1
        tmp_test_in[0, :n_sub_tok] = tmp_i_idx + [SOS_token]
        
        gen_ids = seq_cnn_network.infer(
            tmp_test_in[:, :n_sub_tok], seq_length)
        
        gen_phrase = bpe.bp_decode(
            gen_ids.numpy()[0], idx2subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        gen_tokens = gen_phrase.split(" ")
        num_tokens = len(gen_tokens)
        sos_index  = min([x for x in range(
            num_tokens) if gen_tokens[x] == "SOS"])
        gen_output = " ".join(gen_tokens[sos_index:])
        del gen_tokens, n_sub_tok, num_tokens
        return {"bot_reply": str(gen_output).encode("utf-8")}

