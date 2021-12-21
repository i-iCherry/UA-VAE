
import tensorflow as tf
import torch

# focal loss
def focal_loss(labels, y_pred, alpha=0.25, gamma=2):

    down = 1e-8
    up = 1.0 - 1e-8
    y_pred = tf.clip_by_value(y_pred, down, up)
    loss = -labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.math.log(y_pred) - \
           (1 - labels) * alpha * (y_pred ** gamma) * tf.math.log(1 - y_pred)

    return loss


## gated attention based multi-instance pooling layer
class MIL_gated_attention(tf.keras.layers.Layer):

    def __init__(self, d_model):

        super(MIL_gated_attention, self).__init__()

        self.w1 = tf.keras.layers.Dense(d_model)
        self.w2 = tf.keras.layers.Dense(d_model)
        self.w3 = tf.keras.layers.Dense(d_model)


    def call(self, x):

        # linear projection
        alpha = tf.tanh(self.w1(x))

        # gate mechanism
        gate = tf.nn.sigmoid(self.w2(x))
        alpha = self.w3(tf.multiply(alpha, gate))

        # attention weights
        attention_weights = tf.nn.softmax(alpha)

        # output
        output = tf.multiply(x, attention_weights)
        output = tf.reduce_mean(output, axis=-1)

        return output, attention_weights


# multi-head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):

        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    # scaled dot product attention
    def scaled_dot_product_attention(self, q, k, v):

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


# class MultiHeadAttention(nn.Module):
#     r"""
#     ## Multi-Head Attention Module
#     This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
#     $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
#     In simple terms, it finds keys that matches the query, and get the values of
#      those keys.
#     It uses dot-product of query and key as the indicator of how matching they are.
#     Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
#     This is done to avoid large dot-product values causing softmax to
#     give very small gradients when $d_k$ is large.
#     Softmax is calculate along the axis of of the sequence (or time).
#     """
#
#     def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True,
#                  mask_type: str = 'softmax'):
#         """
#         * `heads` is the number of heads.
#         * `d_model` is the number of features in the `query`, `key` and `value` vectors.
#         """
#
#         super().__init__()
#
#         # Number of features per head
#         self.d_k = d_model // heads
#         # Number of heads
#         self.heads = heads
#
#         # These transform the `query`, `key` and `value` vectors for multi-headed attention.
#         self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
#
#         # Softmax for attention along the time dimension of `key`
#         if mask_type == 'softmax':
#             self.selector = nn.Softmax(dim=1)
#         else:
#             raise NotImplemented()
#
#         # Output layer
#         self.output = nn.Linear(d_model, d_model)
#         # Dropout
#         self.dropout = nn.Dropout(dropout_prob)
#         # Scaling factor before the softmax
#         self.scale = 1 / math.sqrt(self.d_k)
#
#         # We store attentions so that it can used for logging, or other computations if needed
#         self.attn = None
#
#     def get_scores(self, query: torch.Tensor, key: torch.Tensor):
#         """
#         ### Calculate scores between queries and keys
#         This method can be overridden for other variations like relative attention.
#         """
#
#         # Calculate $Q K^\top$
#         return torch.einsum('bihd,bjhd->bijh', query, key)
#
#     def forward(self,
#                 query: torch.Tensor,
#                 key: torch.Tensor,
#                 value: torch.Tensor,
#                 mask: Optional[torch.Tensor] = None):
#         """
#         `query`, `key` and `value` are the tensors that store
#         collection of*query*, *key* and *value* vectors.
#         They have shape `[batch_size, seq_len, d_model]`.
#         `mask` has shape `[batch_size, seq_len, seq_len]` and indicates
#         `mask[b, i, j]` indicates whether for batch `b`,
#         query at position `i` has access to key-value at position `j`.
#         """
#
#         # `query`, `key` and `value`  have shape `[batch_size, seq_len, d_model]`
#         batch_size, seq_len, _ = query.shape
#
#         if mask is not None:
#             # `mask` has shape `[batch_size, seq_len, seq_len]`,
#             # where first dimension is the query dimension.
#             # If the query dimension is equal to $1$ it will be broadcasted
#             assert mask.shape[1] == 1 or mask.shape[1] == mask.shape[2]
#
#             # Same mask applied to all heads.
#             mask = mask.unsqueeze(-1)
#
#         # Prepare `query`, `key` and `value` for attention computation
#         # These will then have shape `[batch_size, seq_len, heads, d_k]`
#         query = self.query(query)
#         key = self.key(key)
#         value = self.value(value)
#
#         # Compute attention scores
#         # Results in a tensor of shape `[batch_size, seq_len, seq_len, heads]`
#         scores = self.get_scores(query, key)
#
#         # Scale scores
#         scores *= self.scale
#
#         # Apply mask
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#
#         # softmax attention along the key sequence dimension
#         attn = self.selector(scores)
#
#         # Apply dropout
#         attn = self.dropout(attn)
#         # Multiply by values
#
#         x = torch.einsum("bijh,bjhd->bihd", attn, value)
#         # Save attentions for any other calculations
#
#         self.attn = attn.detach()
#
#         # Concatenate multiple heads
#         x = x.reshape(batch_size, seq_len, -1)
#
#         # Output layer
#         return self.output(x)