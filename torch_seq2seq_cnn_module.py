
import torch

# Custom Layer Normalisation to make program run #
# faster by preventing transpose operation.      #
def layer_normalisation(
    x, bias, scale, dim=1, eps=1.0e-6):
    x_mean = torch.mean(x, dim, True)
    x_var  = torch.mean(
        torch.square(x - x_mean), dim, True)
    x_norm = (x - x_mean) * torch.rsqrt(x_var + eps)
    return (x_norm * scale) + bias

def CausalConv1d(
    in_channels, out_channels, 
    kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return torch.nn.Conv1d(
        in_channels, out_channels, kernel_size, 
        padding=pad, dilation=dilation, **kwargs)

class Seq2Seq_CNN_Network(torch.nn.Module):
    def __init__(
        self, model_dim, n_layers, 
        enc_vocab_size, dec_vocab_size, n_stacks=1, 
        kernel_size=3, p_drop=0.1, attn_type="add_attn", **kwargs):
        super(Seq2Seq_CNN_Network, self).__init__(**kwargs)
        
        self.prob_drop = p_drop
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        
        self.n_layers  = n_layers
        self.n_stacks  = n_stacks
        self.attn_type = attn_type
        
        self.enc_embed_layer = torch.nn.Embedding(
            self.enc_vocab_size, self.model_dim)
        self.dec_embed_layer = torch.nn.Embedding(
            self.dec_vocab_size, self.model_dim)
        if torch.cuda.is_available():
            self.enc_embed_layer = self.enc_embed_layer.cuda()
            self.dec_embed_layer = self.dec_embed_layer.cuda()
        
        self.l_enc_bias  = torch.nn.Parameter(torch.zeros(
            self.n_stacks, self.n_layers, self.model_dim))
        self.l_enc_scale = torch.nn.Parameter(torch.ones(
            self.n_stacks, self.n_layers, self.model_dim))
        self.l_dec_bias  = torch.nn.Parameter(torch.zeros(
            self.n_stacks, self.n_layers, self.model_dim))
        self.l_dec_scale = torch.nn.Parameter(torch.ones(
            self.n_stacks, self.n_layers, self.model_dim))
        
        enc_cnn_layer_stack = []
        dec_cnn_layer_stack = []
        for n_stack in range(self.n_stacks):
            enc_cnn_layer_list = []
            dec_cnn_layer_list = []
            for n_layer in range(self.n_layers):
                dilation = 2**n_layer
                
                tmp_enc_cnn_layer = CausalConv1d(
                    self.model_dim, self.model_dim, 
                    self.kernel_size, dilation=dilation)
                tmp_dec_cnn_layer = CausalConv1d(
                    self.model_dim, self.model_dim, 
                    self.kernel_size, dilation=dilation)
                
                if torch.cuda.is_available():
                    tmp_enc_cnn_layer = tmp_enc_cnn_layer.cuda()
                    tmp_dec_cnn_layer = tmp_dec_cnn_layer.cuda()
                
                enc_cnn_layer_list.append(tmp_enc_cnn_layer)
                dec_cnn_layer_list.append(tmp_dec_cnn_layer)
            
            enc_cnn_layer_stack.append(enc_cnn_layer_list)
            dec_cnn_layer_stack.append(dec_cnn_layer_list)
        
        self.enc_cnn_layers  = enc_cnn_layer_stack
        self.dec_cnn_layers  = dec_cnn_layer_stack
        
        # Attention Mechanism. #
        if attn_type is not None:
            self.q_layer = torch.nn.Linear(
                self.model_dim, self.model_dim)
            self.k_layer = torch.nn.Linear(
                self.model_dim, self.model_dim)
            self.v_layer = torch.nn.Linear(
                self.model_dim, self.model_dim)
            
            if torch.cuda.is_available():
                self.q_layer = self.q_layer.cuda()
                self.k_layer = self.k_layer.cuda()
                self.v_layer = self.v_layer.cuda()
            
            if attn_type == "add_attn":
                self.a_layer = \
                    torch.nn.Linear(self.model_dim, 1)
                if torch.cuda.is_available():
                    self.a_layer = self.a_layer.cuda()
        
        # Fully Connected layer. #
        self.fc_layer = torch.nn.Linear(
            2*self.model_dim, self.dec_vocab_size)
        if torch.cuda.is_available():
            self.fc_layer = self.fc_layer.cuda()
    
    def encode(self, x, p_drop=0.1):
        relu = torch.nn.ReLU()
        dropout = torch.nn.Dropout(p=p_drop)
        
        # Embedding layer. #
        x_embedding = self.enc_embed_layer(x)
        
        # Transpose the inputs as PyTorch uses channel first. #
        x_stack_input = x_embedding
        x_stack_input = \
            torch.transpose(x_stack_input, 1, 2)
        
        # Convolutional layer. #
        for n_stack in range(self.n_stacks):
            x_cnn_input = x_stack_input
            
            cnn_stack = self.enc_cnn_layers[n_stack]
            for n_layer in range(self.n_layers):
                dilation = 2**n_layer
                pad_causal = (self.kernel_size - 1) * dilation
                
                # CNN Layer. #
                x_cnn_output = relu(
                    cnn_stack[n_layer](x_cnn_input))
                x_cnn_causal = x_cnn_output[:, :, :-pad_causal]
                
                # Layer Normalization. #
                tmp_bias  = torch.unsqueeze(
                    self.l_enc_bias[n_stack][n_layer], axis=1)
                tmp_scale = torch.unsqueeze(
                    self.l_enc_scale[n_stack][n_layer], axis=1)
                
                x_layer_norm = layer_normalisation(
                    x_cnn_causal, tmp_bias, tmp_scale)
                
                # Residual layer. #
                x_residual  = x_cnn_input + x_layer_norm
                x_cnn_input = x_residual
            
            # Residual connection. #
            x_stack_output = dropout(
                x_stack_input + x_residual)
            x_stack_input  = x_stack_output
        return torch.transpose(x_stack_output, 1, 2)
    
    def decode(self, x_stack_encode, x, p_drop=0.1):
        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=2)
        dropout = torch.nn.Dropout(p=p_drop)
        
        # Embedding layer. #
        rsqrt_model = torch.rsqrt(torch.tensor(
            self.model_dim, dtype=torch.float32))
        x_embedding = self.dec_embed_layer(x)
        
        # Transpose the inputs as PyTorch uses channel first. #
        x_stack_input = x_embedding
        x_stack_input = \
            torch.transpose(x_stack_input, 1, 2)
        
        # Convolutional layer. #
        for n_stack in range(self.n_stacks):
            x_cnn_input = x_stack_input
            
            cnn_stack = self.dec_cnn_layers[n_stack]
            for n_layer in range(self.n_layers):
                dilation = 2**n_layer
                pad_causal = (self.kernel_size - 1) * dilation
                
                # CNN Layer. #
                x_cnn_output = relu(
                    cnn_stack[n_layer](x_cnn_input))
                x_cnn_causal = x_cnn_output[:, :, :-pad_causal]
                
                # Layer Normalization. #
                tmp_bias  = torch.unsqueeze(
                    self.l_dec_bias[n_stack][n_layer], axis=1)
                tmp_scale = torch.unsqueeze(
                    self.l_dec_scale[n_stack][n_layer], axis=1)
                
                x_layer_norm = layer_normalisation(
                    x_cnn_causal, tmp_bias, tmp_scale)
                
                # Residual layer. #
                x_residual  = x_cnn_input + x_layer_norm
                x_cnn_input = x_residual
            
            # Residual connection. #
            x_stack_output = dropout(
                x_stack_input + x_residual)
            x_stack_input  = x_stack_output
        
        
        # Attention Mechanism. #
        x_stack_decode = \
            torch.transpose(x_stack_output, 1, 2)
        
        x_q = self.q_layer(x_stack_decode)
        x_q = x_q * rsqrt_model
        x_k = self.k_layer(x_stack_encode)
        x_v = self.v_layer(x_stack_encode)
        if self.attn_type == "add_attn":
            x_q = torch.unsqueeze(x_q, axis=2)
            x_k = torch.unsqueeze(x_k, axis=1)
            x_add_input = tanh(x_q + x_k)
            
            x_scores = self.a_layer(x_add_input)
            x_scores = torch.squeeze(x_scores, axis=3)
            x_alphas = softmax(x_scores)
        else:
            x_scores = torch.matmul(
                x_q, torch.transpose(x_k, 1, 2))
            x_alphas = softmax(x_scores)
        
        x_attn_outputs = torch.matmul(x_alphas, x_v)
        x_final_output = torch.cat(tuple(
            [x_embedding, x_attn_outputs]), axis=2)
        
        # Output logits. #
        x_logits = self.fc_layer(x_final_output)
        return x_logits
    
    def forward(
        self, x_encode, x_decode, 
        training=None, mask=None):
        if training:
            p_drop = self.prob_drop
        else:
            p_drop = 0.0
        
        x_stack_encode  = self.encode(x_encode, p_drop=p_drop)
        x_output_logits = self.decode(
            x_stack_encode, x_decode, p_drop=p_drop)
        return x_output_logits
    
    def infer(
        self, x_encode, SOS_token, seq_output):
        batch_size  = list(x_encode.size())[0]
        tmp_sos_tok = SOS_token * torch.ones(
            batch_size, 1, dtype=torch.long)
        
        if torch.cuda.is_available():
            tmp_sos_tok = tmp_sos_tok.cuda()
        tmp_outputs = [tmp_sos_tok]
        
        x_stack_encode = self.encode(x_encode, p_drop=0.0)
        for n_seq in range(seq_output):
            tmp_decode = torch.cat(tmp_outputs, axis=1)
            tmp_logits = self.decode(
                x_stack_encode, tmp_decode, p_drop=0.0)
            
            tmp_index = torch.argmax(
                tmp_logits[:, -1, :], axis=1)
            tmp_index = torch.unsqueeze(tmp_index, axis=1)
            tmp_outputs.append(tmp_index)
        return torch.cat(tmp_outputs, axis=1)

def train_step(
    model, sub_batch_sz, 
    x_input, x_decode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    for tmp_opt_group in optimizer.param_groups:
        tmp_opt_group["lr"] = learning_rate
    
    batch_size = x_input.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    avg_batch_loss = 0.0
    
    optimizer.zero_grad()
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_input  = torch.tensor(
            x_input[id_st:id_en, :], dtype=torch.long)
        tmp_decode = torch.tensor(
            x_decode[id_st:id_en, :], dtype=torch.long)
        tmp_output = torch.tensor(
            x_output[id_st:id_en, :], dtype=torch.long)
        
        if torch.cuda.is_available():
            tmp_input  = tmp_input.cuda()
            tmp_decode = tmp_decode.cuda()
            tmp_output = tmp_output.cuda()
        
        output_logits = model(
            tmp_input, tmp_decode, training=True)
        
        tmp_losses = torch.sum(torch.sum(ce_loss_fn(
            torch.transpose(output_logits, 1, 2), tmp_output), 1))
        sub_batch_loss = tmp_losses / batch_size
        
        # Accumulate the gradients. #
        sub_batch_loss.backward()
        avg_batch_loss += sub_batch_loss.item()
    
    # Update using the optimizer. #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip)
    optimizer.step()
    return avg_batch_loss


