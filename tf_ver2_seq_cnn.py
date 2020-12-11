
import tensorflow as tf
from tensorflow.keras import layers

def build_model(
    n_filters, vocab_size, 
    n_layers, max_length, 
    p_drop=0.10, n_stacks=1, kernel_size=3):
    if n_layers <= 0:
        raise ValueError("No. of layers must be positive.")
    if n_stacks <= 0:
        raise ValueError("No. of stacks must be positive.")
    
    b_stack = tf.Variable(tf.random.normal(
        [1, n_stacks, max_length, n_filters], stddev=0.1), 
        name="position_stack_embeddings", trainable=True)
    b_layer = tf.Variable(tf.random.normal(
        [1, n_layers, max_length, n_filters], stddev=0.1), 
        name="position_layer_embeddings", trainable=True)
    x_input = tf.keras.Input(
        shape=(None,), dtype="int32", name="x_input")
    x_length = tf.shape(x_input)[1]
    
    # Embedding layer. #
    embed_layer = layers.Embedding(
        vocab_size, n_filters, name="embedding")
    x_embedding = embed_layer(x_input)
    
    # Convolutional layer. #
    x_stack_input = x_embedding
    for n_stack in range(n_stacks):
        stack_name = "stack_" + str(n_stack+1) + "_"
        x_stack_input = tf.add(
            x_stack_input, b_stack[:, n_stack, :x_length, :])
        
        x_cnn_input = x_stack_input
        for n_layer in range(n_layers):
            cnn_name  = "cnn_layer_" + str(n_layer+1)
            norm_name = "norm_layer_" + str(n_layer+1)
            if n_layer == 0:
                dilation_rate = 1
            else:
                dilation_rate = 2
            
            x_cnn_input = tf.add(
                x_cnn_input, b_layer[:, n_layer, :x_length, :])
            
            x_cnn_output = layers.Conv1D(
                n_filters, kernel_size, 
                dilation_rate=dilation_rate, 
                padding="causal", activation="relu", 
                strides=1, name=stack_name + cnn_name)(x_cnn_input)
            
            # Layer Normalization. #
            x_layer_norm = layers.LayerNormalization(
                    name=stack_name + norm_name)(x_cnn_output)
            
            # Residual layer. #
            x_residual  = x_cnn_input + x_layer_norm
            x_cnn_input = x_residual
        
        # Skip connection. #
        x_skip_output = \
            layers.Dropout(rate=p_drop)(
                x_stack_input + x_residual)
        x_stack_input = x_skip_output
    
    # Output logits. #
    x_logits = layers.Dense(
        vocab_size, name="logits")(x_embedding + x_skip_output)
    return tf.keras.Model(inputs=x_input, outputs=x_logits)

def train_step(
    model, sub_batch_sz, 
    x_input, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_input  = x_input[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(tmp_input, training=True)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
            tot_losses += tmp_losses
        
        # Accumulate the gradients. #
        tmp_gradients = \
            grad_tape.gradient(tmp_losses, model_params)
        acc_gradients = [(acc_grad+grad) for \
            acc_grad, grad in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_cls_loss  = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_cls_loss

def infer(
    model, x_input, seq_output, kernel_size=3):
    len_input = x_input.shape[1]
    
    tmp_outputs = [tf.constant(x_input[:, :kernel_size])]
    for n_seq in range(kernel_size, seq_output):
        tmp_inputs = tf.concat(tmp_outputs, axis=1)
        if n_seq < seq_output:
            tmp_input = tmp_inputs[:, :n_seq]
        else:
            tmp_input = \
                tmp_inputs[:, (n_seq-len_input):n_seq]
        
        tmp_logits = model.predict(tmp_input)
        if n_seq < len_input:
            tmp_index = \
                tf.expand_dims(x_input[:, n_seq], axis=1)
        else:
            tmp_index = tf.random.categorical(
                tmp_logits[:, -1, :], 1, dtype=tf.int32)
        tmp_outputs.append(tmp_index)
    return tf.concat(tmp_outputs, axis=1)

