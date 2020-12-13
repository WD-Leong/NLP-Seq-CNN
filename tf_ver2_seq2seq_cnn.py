import tensorflow as tf
from tensorflow.keras import layers

def build_model(
    n_filters, n_layers, seq_enc, 
    vocab_size_enc, vocab_size_dec, 
    p_drop=0.10, n_stacks=1, kernel_size=3):
    if n_layers <= 0:
        raise ValueError("No. of layers must be positive.")
    if n_stacks <= 0:
        raise ValueError("No. of stacks must be positive.")
    
    x_encode = tf.keras.Input(
        shape=(seq_enc,), dtype="int32", name="x_encode")
    x_decode = tf.keras.Input(
        shape=(None,), dtype="int32", name="x_decode")
    x_input  = [x_encode, x_decode]
    
    # Embedding layer. #
    enc_embed_layer = layers.Embedding(
        vocab_size_enc, n_filters, name="enc_embedding")
    dec_embed_layer = layers.Embedding(
        vocab_size_dec, n_filters, name="dec_embedding")
    
    x_enc_embed = enc_embed_layer(x_encode)
    x_dec_embed = dec_embed_layer(x_decode)
    
    # Encoder Convolutional layer. #
    x_stack_encode = x_enc_embed
    for n_stack in range(n_stacks):
        stack_name = "enc_stack_" + str(n_stack+1) + "_"
        
        x_cnn_input = x_stack_encode
        for n_layer in range(n_layers):
            cnn_name  = "cnn_layer_" + str(n_layer+1)
            norm_name = "norm_layer_" + str(n_layer+1)
            if n_layer == 0:
                dilation_rate = 1
            else:
                dilation_rate = 2
            
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
                x_stack_encode + x_residual)
        x_stack_encode = x_skip_output
    
    # Decoder Convolutional layer. #
    x_stack_decode = x_dec_embed
    for n_stack in range(n_stacks):
        stack_name = "dec_stack_" + str(n_stack+1) + "_"
        
        x_cnn_input = x_stack_decode
        for n_layer in range(n_layers):
            cnn_name  = "cnn_layer_" + str(n_layer+1)
            norm_name = "norm_layer_" + str(n_layer+1)
            if n_layer == 0:
                dilation_rate = 1
            else:
                dilation_rate = 2
            
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
                x_stack_decode + x_residual)
        x_stack_decode = x_skip_output
    
    # Encoder Decoder Attention. #
    x_q = layers.Dense(
        n_filters, name="x_q")(x_stack_decode)
    x_k = layers.Dense(
        n_filters, name="x_k")(x_stack_encode)
    x_v = layers.Dense(
        n_filters, name="x_v")(x_stack_encode)
    
    x_softmax = tf.nn.softmax(
        tf.matmul(x_q, x_k, transpose_b=True))
    x_att_out = tf.matmul(x_softmax, x_v)
    x_out_conc = tf.concat([x_stack_decode, x_att_out], axis=2)
    
    # Output logits. #
    x_logits = layers.Dense(
        vocab_size_dec, name="logits")(x_out_conc)
    return tf.keras.Model(inputs=x_input, outputs=x_logits)

def train_step(
    model, sub_batch_sz, x_encode, x_decode, x_output, 
    optimizer, learning_rate=1.0e-3, gradient_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
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
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_decode = x_decode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(
                [tmp_encode, tmp_decode], training=True)
            
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
        tf.clip_by_global_norm(acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_cls_loss

def infer(
    model, x_encode, SOS_token, seq_output, kernel_size=3):
    batch_size = tf.shape(x_encode)[0]
    
    tmp_outputs = [\
        SOS_token * tf.ones([batch_size, 1], dtype=tf.int32)]
    for n_seq in range(seq_output):
        tmp_decode = tf.concat(tmp_outputs, axis=1)
        tmp_input  = [x_encode, tmp_decode]
        
        tmp_logits = model.predict(tmp_input)
        tmp_index  = tf.argmax(
            tmp_logits[:, -1, :], axis=1, output_type=tf.int32)
        tmp_outputs.append(tf.expand_dims(tmp_index, axis=1))
    return tf.concat(tmp_outputs, axis=1)