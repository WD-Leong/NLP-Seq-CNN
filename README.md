# NLP Language Model using Dilated Causal Convolutional Networks 
This repository includes the codes that I explored for language modelling using a 1D Dilated Causal Convolutional Networks. It is inspired from the [WaveNet Model](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), which I modified and applied to the Language Modelling task. By applying dilated causal convolutional networks (CNN), the model increases the receptive field exponentially to allow it to model long range dependencies and sequences. The first layer has a dilation rate of 1, but subsequent layers have a dilation rate of 2. 

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
to train the model. Training can be done in parallel, which is a desirable property. To perform inference, the model runs auto-regressively on the seed input, or the current output sequence. To perform inference, run the code
```
python reddit_jokes_seq_cnn_test.py
```
Instead of using temperature, `tf.random.categorical` function is applied on the logits directly to introduce diversity in the inferred joke. Depending on the output sequence length, the inference can take some time.

## Dilated Convolutional Networks
Following 

![WaveNet's Dilated 1D Convolutional Network](https://github.com/WD-Leong/NLP-Seq-CNN/WaveNet Dilated Convolution.jpg)

## Outputs
After training a model with 256 filters, 4 layers and 2 stacks on Reddit jokes with 30 tokens or less for 20000 iterations, some outputs are provided in this section.
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
