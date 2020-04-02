import math
import tensorflow as tf

# Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, name = 'multi_head_attention'):
        super().__init__(name = name)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name = 'query')
        self.wk = tf.keras.layers.Dense(d_model, name = 'key')
        self.wv = tf.keras.layers.Dense(d_model, name = 'value')

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'mha_dropout')
        self.dense = tf.keras.layers.Dense(d_model, name = 'mha_output')

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.
        
        Returns:
            output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b = True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., seq_len_q, seq_len_k)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (As claimed in the RoBERTa implementation.)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def gelu(x):
    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + tf.math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + tf.math.erf(x / math.sqrt(2.0)))

act_funcs = {'gelu': gelu, 'relu': tf.nn.relu}


# Pointwise Feed Forward Network
def point_wise_feed_forward_network(d_model, dff, hidden_act):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation = act_funcs[hidden_act],
            name = 'ff_hidden'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, name = 'ff_output')  # (batch_size, seq_len, d_model)
    ], name = 'ff_network')


# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, layer_num):
        super().__init__(name = 'encoder_layer_{:02d}'.format(layer_num))

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff, hidden_act)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_2')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_1')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_2')

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

def loss_function(real_emot, pred_emot):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')
    loss_ = scce(real_emot, pred_emot)
    return loss_


class EmoBERT(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, vocab_size, num_emotions):
        super().__init__(name = 'emo_bert')

        self.padding_idx = 1

        # Embedding layers
        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

        # Encoder layers
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

        # Output layers
        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')
        self.hidden_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(num_emotions, name = 'output_layer')

    def call(self, x, training, mask):
        # x.shape == (batch_size, seq_len)

        seq_len = tf.shape(x)[1]

        # Add word embedding and position embedding.
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, tf.shape(x))
        x = self.word_embeddings(x)  # (batch_size, seq_len, d_model)
        x += self.pos_embeddings(pos)

        x = self.layernorm(x)
        x = self.dropout(x, training = training)

        # x.shape == (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # Compute the attention scores
        projected = self.attention_layer(x)  # (batch_size, seq_len, d_model)
        scores = tf.nn.softmax(tf.squeeze(self.attention_v(projected), 2))
        scores = tf.expand_dims(scores, 1)  # (batch_size, 1, seq_len)

        # x.shape == (batch_size, d_model)
        x = tf.squeeze(tf.matmul(scores, x), 1)

        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x  # (batch_size, num_emotions)