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

![WaveNet's Dilated 1D Convolutional Network](WaveNet_Dilated_Convolution.JPG)

Fig 1.: WaveNet's Dilated Convolutional Network (source: [WaveNet](https://arxiv.org/pdf/1609.03499.pdf))

## Outputs
A model with 256 filters, 4 layers and 2 stacks on Reddit jokes with a maximum of 30 tokens was trained on an Nvidia Quadro P1000 4GB Graphics Card for 20000 iterations. Some of the model's output are provided in this section.
```
Input Phrase:
did you hear
Generated Phrase:
did you hear about the restaurant on the moon ? yeah , great food but no atmosphere . EOS

Input Phrase:
what is the difference
Generated Phrase:
what is the difference between a sharply dressed man on a unicycle and a poorly dressed man on a bicycle ?? attire EOS

Input Phrase:
why did
Generated Phrase:
why did the cat stop singing ? because it was out of tuna half a note . EOS
```
Overall, it was observed that the Sequence-CNN model is able to model much longer sequences, but its performance would not be as good as the GPT model if the hidden size of both models were the same.

## Extension to Sequence-to-Sequence Models
Inspired by [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) paper, a much simplified Sequence-to-Sequence CNN model using dilated causal convolution is developed here. Unlike Transformers, the attention mechanism is only applied once at the final layer of the Encoder and Decoder outputs. It is conceptually simple and bears some resemblance to RNN/LSTM networks. Similar to the implementation of sequence dilated causal convolutional neural networks, no positional embedding is applied in this implementation.

![Sequence-to-Sequence Dilated Convolutional Architecture](Seq2Seq_CNN_Architecture.jpg)

Fig 2.: Sequence-to-Sequence Architecture using Dilated Convolutional Network (Diagram modified from [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper)

The Sequence-to-Sequence Dilated Convolutional Network is applied on the movie dialogue dataset. To process the data, run
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
with the input phrase of your choice.
