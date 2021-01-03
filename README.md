# NLP Language Modelling using Dilated Causal Convolutional Networks 
This repository includes the codes that was explored for language modelling using a 1D Dilated Causal Convolutional Networks. It is inspired from the [WaveNet Model](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), which was modified and applied to the Language Modelling task here. By applying dilated causal convolutional networks (CNN), the model increases the receptive field exponentially to allow it to model long range dependencies and sequences. The first layer has a dilation rate of 1, but subsequent layers have a dilation rate of 2. 

This repository includes a code to train the data on the [Reddit Jokes](https://github.com/taivop/joke-dataset) dataset. To train the model, first process the Reddit data by running
```
python process_reddit_jokes.py
```
if word tokens are used, or
```
python process_reddit_jokes_subword.py
```
if sub-word tokens are used. After processing the data, run
```
python reddit_jokes_seq_cnn_train.py
```
to train the model. The CNN structure allows training to be optimized on a GPU. To perform inference, the model runs auto-regressively on the seed input, or the current output sequence. To perform inference, run the code
```
python reddit_jokes_seq_cnn_test.py
```
Instead of using temperature, `tf.random.categorical` function is applied on the logits directly to introduce diversity in the inferred joke. Depending on the output sequence length, the inference can take some time. In the processing, the score assigned to the joke is categorized into 3 classes - bad, ok and good - to study its effect on the quality of the jokes generated.

## Dilated Convolutional Networks
The dilated convolutional neural network applied in the [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper allows it to cover thousands of timesteps, making it suitable to generate synthetic utterances. Unlike the [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) paper, no position embedding is applied in this implementation.

![WaveNet's Dilated 1D Convolutional Network](Dilated_Convolution.jpg)

Fig 1: WaveNet's Dilated Convolutional Network (source: [WaveNet](https://arxiv.org/pdf/1609.03499.pdf))

## Outputs
A model with 256 filters, 4 layers, 2 stacks and a convolution width (`kernel_size`) of 3 on Reddit jokes with a maximum of 30 word tokens was trained on an Nvidia Quadro P1000 4GB Graphics Card for 20000 iterations. Some of the model's output are provided in this section.
```
Input Phrase:
bad_joke
Generated Phrase:
bad_joke what do you call a hugh fish boat in a hairs ? a business . EOS

Input Phrase:
ok_joke
Generated Phrase:
ok_joke what do you get when you cross a joke with a rhetorical question ? ... EOS

Input Phrase:
good_joke
Generated Phrase:
good_joke " master yoda , are we on the right track ?" " off course , we are ." EOS
```
Overall, it was observed that the Sequence-CNN model is able to model much longer sequences, but its performance would not be as good as the GPT model if the hidden size of both models were the same.

## Dilated CNN Models with Attention

An extention to the Sequence CNN model is explored to incorporate attention. In this case, it is first observed that the different layers correspond to sequence outputs whose receptive fields are of different lengths. Hence, inspired by the [Compressive Transformer](https://arxiv.org/abs/1911.05507), the attention mechanism is done across the outputs of the different stacks/layers, allowing the model to combine outputs across different receptive fields. As the total number of stacks/layers is generally much lower than the sequence length, this operation is gentler on the hardware memory while maintaining an acceptable degree of performance.

![Dilated CNN Models with Attention](Dilated_Convolution_Attention.jpg)
Fig. 2: Model Architecture of Dilated CNN Model with Attention Mechanism (Diagram modified from that in [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper)

The Sequence CNN module is provided in the `tf_ver2_seq_cnn_attn.py` module. As the model parameters of Sequence CNN and Sequence CNN Attention are similar, changing the import module 
```
import tf_ver2_seq_cnn as tf_model
```
to
```
import tf_ver2_seq_cnn_attn as tf_model
```
would allow the `reddit_jokes_seq_cnn_train.py` and `reddit_jokes_seq_cnn_test.py` to train and infer respectively using the intended model.

Some of the Sequence CNN Attention Model outputs for the Reddit joke dataset are provided below:
```
Input Phrase:
bad_joke
Generated Phrase:
bad_joke what do you call a pile of kittens ? a meowntain EOS

Input Phrase:
ok_joke
Generated Phrase:
ok_joke a hamburger walks into a bar and orders a salad the bartender says " sorry , we don t serve food here ." EOS

Input Phrase:
good_joke
Generated Phrase:
good_joke what kind of pants does mario wear ? denim denim denim EOS
```

For the movie dialogue dataset, run the following scripts in order:
```
python process_movie_dialogue.py
python dialogue_seq_cnn_train.py
```
to train the model. Run
```
python dialogue_seq_cnn_test.py
```
to perform inference. 

Some of sample outputs for the Sequence CNN Attention Model for the movie dialogue dataset are provided below:
```
Input Phrase:
hi
Generated Response:
hi EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Input Phrase:
how much does it cost
Generated Response:
two thousand francs EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Input Phrase:
where did he come from
Generated Response:
i don t know he just appeared as magic EOS PAD PAD PAD PAD PAD

Input Phrase:
who is it
Generated Response:
it s me EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Input Phrase:
what time is it
Generated Response:
eight o clock you got to go EOS PAD PAD PAD PAD PAD PAD PAD PAD
```

## Extension to Sequence-to-Sequence Models
Inspired by [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) paper, a much simplified Sequence-to-Sequence CNN model using dilated causal convolution is developed here. Unlike Transformers, the attention mechanism is only applied once at the final layer of the Encoder and Decoder outputs. It is conceptually simple and bears some resemblance to RNN/LSTM networks. Similar to the implementation of sequence dilated causal convolutional neural networks, no positional embedding is applied in this implementation.

![Sequence-to-Sequence Dilated Convolutional Architecture](Seq2Seq_CNN_Architecture.jpg)

Fig. 3: Sequence-to-Sequence Architecture using Dilated Convolutional Network (Diagram modified from [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper)

The Sequence-to-Sequence Dilated Convolutional Network is applied on the [movie dialogue dataset](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The pre-processing of the data follow this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. First, process the data by running
```
python process_movie_dialogue.py
```
followed by
```
python dialogue_seq2seq_cnn_train.py
```
to train the model. To perform inference, run
```
python dialogue_seq2seq_cnn_test.py
```
with the input phrase of your choice. In the scripts, a convolutional width (`kernel_size`) of 5 is set. For subword tokens, run
```
python process_movie_dialogue_subwords.py
python dialogue_subword_seq2seq_cnn_train.py
```
to train the model, and
```
python dialogue_subword_seq2seq_cnn_test.py
```
to perform inference.

The Sequence-to-Sequence CNN model is trained for 20000 iterations using the same hardware (Nvidia P1000 4GB Graphics Card). Some of the outputs using word tokens is provided below:
```
Input Phrase:
good morning
Generated Phrase:
good morning EOS

Input Phrase:
who are you
Generated Phrase:
i m the one who wanted to study it EOS

Input Phrase:
where are you going
Generated Phrase:
i m going to california EOS

Input Phrase:
where are we going
Generated Phrase:
we re not going anywhere EOS

Input Phrase:
how are you
Generated Phrase:
i m fine EOS
```

For the subword tokens, the The Sequence-to-Sequence CNN model is trained for 50000 iterations. Some of the sample outputs of the model is provided below (additional `PAD` tokens after `EOS` are removed):
```
Input Phrase:
where are you going ?
Generated Phrase:
SOS i ' m a little nervous , so ... EOS

Input Phrase:
how are you ?
Generated Phrase:
SOS fine . EOS

how much ?
Generated Phrase:
SOS $ 50 ? EOS

Input Phrase:
what time is it ?
Generated Phrase:
SOS eight o ' clock . EOS

Input Phrase:
when are we leaving ?
Generated Phrase:
SOS tomorrow morning , before the store opening . EOS
```
