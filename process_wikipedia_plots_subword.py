
import time
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

print("Loading the data.")
start_tm = time.time()

tmp_file = "C:/Users/admin/Desktop/Data/wikipedia_plots/plots"
with open(tmp_file, "rb") as tmp_file_open:
    raw_data = tmp_file_open.readlines()

tmp_plot = ""
tmp_data = []
for tmp_line in raw_data:
    tmp_line = tmp_line.decode("utf-8")
    tmp_plot += " " + tmp_line
    
    if tmp_line == "<EOS>\n":
        tmp_data.append(tmp_plot)
        tmp_plot = ""

# Extract the data. #
max_len = 200
print("Total of", str(len(tmp_data)), "plots loaded.")

# Process the data. #
tmp_plots_filtered = []

w_counter = Counter()
for tmp_plot in tmp_data:
    tmp_tokens = [
        x for x in word_tokenizer(tmp_plot.lower()) if x != ""]
    
    if len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        tmp_plots_filtered.append(tmp_plot)
    del tmp_tokens

print("Total of", str(len(tmp_plots_filtered)), "plots filtered.")
del tmp_data

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join([x+" " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", str(len(word_counts)), "words.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

n_iters = 500
vocab_size = 10000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab_list = tuple_out[0]
idx2subword = tuple_out[1]
subword2idx = tuple_out[2]

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      str(len(subword_vocab_list)), "sub-word tokens")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()

new_plots = []
for tmp_plot in tmp_plots_filtered:
    tmp_plot_sw = bpe.bp_encode(
        tmp_plot, subword_vocab_list, subword2idx)
    new_plots.append(tmp_plot_sw)

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Save the data. #
print("Saving the file.")

tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/wikipedia_plot_subword.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(new_plots, tmp_file_save)
    pkl.dump(idx2subword, tmp_file_save)
    pkl.dump(subword2idx, tmp_file_save)
