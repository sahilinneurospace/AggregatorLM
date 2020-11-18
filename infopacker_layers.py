from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf
from tensorflow.python.keras import backend as K
import keras
from recurrent import LSTMCell

def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.

  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class InfoPacker(tf.keras.layers.Layer):
  """Self Information extraction:
    Three sources of information for a token:
        1.  Purely context information: Extracted by running Bi-Directional LSTM over value vector sequence
        2.  Purely individual information: Value vector of the token
        3.  Token-directed context information: Extracted using following procedure-
        a.  Build information extractor from the query vector of the token
        b.  Use the info extractor to extract info vectors from value vectors of the context tokens
        c.  Take dot product between query, key vector and Softmax to get weights
        d.  Get weighted sum of info vectors
    
    Self information-extraction operations:
        
        V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
        Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
        K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
        Information_extractors:[BFNHI] = einsum(‘BFNQ,QHI->BFNHI’, Q, WI)
        Pure_context_information:[BTNH] = dot(BiLSTM(V), Wc)
        Information_extracted:[BNFTI] = einsum(‘BTNH,BFNHI->BNFTI’, V, Information_extractors)
        Attention_scores:[BNFT] = einsum('BTNH,BFNH->BNFT', K, Q) / sqrt(H)
        Attention_probs:[BNFT] = softmax(Attention_scores)
        Directed_context_information:[BFNI] = einsum(‘BNFT,BNFTI->BFNI’, Attention_probs, Information_extracted)
        Complete_information[BFNX] = Concatenate(V, Pure_ context_information, Directed_context_information)
        Output[BFD] = einsum(‘BFNX,NXD->BFD’, Complete_information, Wo)
        Note: F=T, I=H, Q=H, X=3H
  """

  def __init__(self,
               base_vector_size=128,
               num_info_heads=12,
               size_per_head=64,
               infopacker_hidden_dropout_prob=0.0,
               initializer_range=0.02,
               **kwargs):
    super(InfoPacker, self).__init__(**kwargs)
    self.base_vector_size = base_vector_size
    self.num_info_heads = num_info_heads
    self.size_per_head = size_per_head
    self.infopacker_hidden_dropout_prob = infopacker_hidden_dropout_prob
    self.initializer_range = initializer_range

  def build(self, input_shapes):
    """Implements build() for the layer."""
    self.value_kernel = self.add_weight("value_kernel", 
                                        shape=[self.base_vector_size, self.num_info_heads, self.size_per_head],
                                        initializer=get_initializer(self.initializer_range),
                                        dtype=tf.float32,
                                        trainable=True)
    self.value_bias = self.add_weight("value_bias",
                                      shape=[self.num_info_heads, self.size_per_head],
                                      initializer=get_initializer(self.initializer_range),
                                      dtype=tf.float32,
                                      trainable=True)
    self.query_kernel = self.add_weight("query_kernel", 
                                        shape=[self.base_vector_size, self.num_info_heads, self.size_per_head],
                                        initializer=get_initializer(self.initializer_range),
                                        dtype=tf.float32,
                                        trainable=True)
    self.query_bias = self.add_weight("query_bias",
                                      shape=[self.num_info_heads, self.size_per_head],
                                      initializer=get_initializer(self.initializer_range),
                                      dtype=tf.float32,
                                      trainable=True)
    self.key_kernel = self.add_weight("key_kernel", 
                                      shape=[self.base_vector_size, self.num_info_heads, self.size_per_head],
                                      initializer=get_initializer(self.initializer_range),
                                      dtype=tf.float32,
                                      trainable=True)
    self.key_bias = self.add_weight("key_bias",
                                    shape=[self.num_info_heads, self.size_per_head],
                                    initializer=get_initializer(self.initializer_range),
                                    dtype=tf.float32,
                                    trainable=True)
    self.pure_context_kernel = self.add_weight("pure_context_kernel", 
                                               shape=[2 * self.size_per_head, self.size_per_head],
                                               initializer=get_initializer(self.initializer_range),
                                               dtype=tf.float32,
                                               trainable=True)
    self.pure_context_bias = self.add_weight("pure_context_bias",
                                             shape=[self.size_per_head],
                                             initializer=get_initializer(self.initializer_range),
                                             dtype=tf.float32,
                                             trainable=True)
    self.output_kernel = self.add_weight("output_kernel", 
                                         shape=[self.num_info_heads, 3*self.size_per_head, self.base_vector_size],
                                         initializer=get_initializer(self.initializer_range),
                                         dtype=tf.float32,
                                         trainable=True)
    self.output_bias = self.add_weight("output_bias",
                                    shape=[self.base_vector_size],
                                    initializer=get_initializer(self.initializer_range),
                                    dtype=tf.float32,
                                    trainable=True)
    self.directed_context_info_bias = self.add_weight("directed_context_info_bias",
                                                      shape=[self.size_per_head],
                                                      initializer=get_initializer(self.initializer_range),
                                                      dtype=tf.float32,
                                                      trainable=True)
    self.infopacker_dropout = tf.keras.layers.Dropout(rate=self.infopacker_hidden_dropout_prob)

    super(InfoPacker, self).build(input_shapes)

  def __call__(self, input_tensor, info_extractor_builder, rnn_fwd_cell, rnn_bwd_cell, info_mask=None, **kwargs):
    inputs = [input_tensor, info_extractor_builder, rnn_fwd_cell, rnn_bwd_cell, info_mask]
    return super(InfoPacker, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_tensor, info_extractor_builder, rnn_fwd_cell, rnn_bwd_cell, attention_mask = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

    value_tensor = tf.einsum('BFD,DNH->BFNH', input_tensor, self.value_kernel) + self.value_bias

    query_tensor = tf.einsum('BFD,DNH->BFNH', input_tensor, self.query_kernel) + self.query_bias

    key_tensor = tf.einsum('BFD,DNH->BFNH', input_tensor, self.key_kernel) + self.key_bias

    info_extractor = tf.einsum('BFNQ,QHI->BFNHI', query_tensor, info_extractor_builder)

    info_extracted = tf.einsum('BTNH,BFNHI->BNFTI', value_tensor, info_extractor)

    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor) 

    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)

    attention_probs = self.attention_probs_dropout(attention_probs)

    directed_context_info = tf.einsum('BNFT,BNFTI->BFNI', attention_probs, info_extracted) + \
                            self.directed_context_info_bias

    context_rnn_outputs = tf.compat.v1.nn.bidirectional_dynamic_rnn(rnn_fwd_cell, rnn_bwd_cell, value_tensor)

    pure_context_info = tf.einsum('BFNC,CH->BFNH', tf.concat(list(context_rnn_outputs), axis=-1),
                                  self.pure_context_kernel) + self.pure_context_bias
    
    complete_info = tf.concat([value_tensor, pure_context_info, directed_context_info], axis=-1)

    output_tensor = tf.einsum("BFNX,NXD->BFD", complete_info, self.output_kernel) + self.output_bias

    return output_tensor


  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'base_vector_size': self.base_vector_size,
        'num_attention_heads': self.num_attention_heads,
        'size_per_head': self.size_per_head,
        'infopacker_hidden_dropout_prob': self.infopacker_hidden_dropout_prob,
        'initializer_range': self.initializer_range
    })
    return config


class AggregatorBlock(tf.keras.layers.Layer):

  def __init__(self,
               base_vector_size=128,
               num_info_heads=12,
               size_per_head=64,
               infopacker_hidden_dropout_prob=0.0,
               initializer_range=0.02,
               info_dropout_prob=0.0,
               **kwargs):
    super(AggregatorBlock, self).__init__(**kwargs)
    self.base_vector_size = base_vector_size
    self.num_info_heads = num_info_heads
    self.size_per_head = size_per_head
    self.infopacker_hidden_dropout_prob = infopacker_hidden_dropout_prob
    self.initializer_range = initializer_range
    self.info_dropout_prob = info_dropout_prob

  def build(self, input_shapes):
    
    self.infopacker_layer = InfoPacker(
        base_vector_size=self.base_vector_size,
        num_info_heads=self.num_info_heads,
        size_per_head=self.size_per_head,
        infopacker_hidden_dropout_prob=self.infopacker_hidden_dropout_prob,
        initializer_range=self.initializer_range)

    self.infopacker_dropout = tf.keras.layers.Dropout(
        rate=self.info_dropout_prob)

    self.infopacker_layer_norm = tf.keras.layers.LayerNormalization(
        name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
        dtype=tf.float32)

    self.rnn_forward_cell = LSTMCell(self.size_per_head)

    self.rnn_backward_cell = LSTMCell(self.size_per_head)

    super(AggregatorBlock, self).build(input_shapes)

  def __call__(self, input_tensor, info_extractor_builder, info_mask=None, **kwargs):
    inputs = [input_tensor, info_extractor_builder, info_mask]
    return super(AggregatorBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    
    input_tensor, info_extractor_builder, attention_mask = inputs[0], inputs[1], inputs[2]
    infopacker_output = self.infopacker_layer(Input_tensor, info_extractor_builder, self.rnn_forward_cell, self.rnn_backward_cell, attention_mask)
    infopacker_output = self.infopacker_dropout(infopacker_output)
    infopacker_output = self.infopacker_layer_norm(infopacker_output)

    return infopacker_output

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'base_vector_size': self.base_vector_size,
        'num_attention_heads': self.num_attention_heads,
        'size_per_head': self.size_per_head,
        'infopacker_hidden_dropout_prob': self.infopacker_hidden_dropout_prob,
        'initializer_range': self.initializer_range,
        'infopacker_dropout_prob': self.infopacker_dropout_prob
    })
    return config


class Aggregator(tf.keras.layers.Layer):

  def __init__(self,
               num_hidden_layers=2,
               base_vector_size=128,
               num_info_heads=12,
               size_per_head=64,
               infopacker_hidden_dropout_prob=0.0,
               initializer_range=0.02,
               infopacker_dropout_prob=0.0,
               **kwargs):
    super(Aggregator, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.base_vector_size = base_vector_size
    self.num_info_heads = num_info_heads
    self.size_per_head = size_per_head
    self.infopacker_hidden_dropout_prob = infopacker_hidden_dropout_prob
    self.initializer_range = initializer_range
    self.infopacker_dropout_prob = infopacker_dropout_prob

  def build(self, input_shapes):
    self.infopacker_layers = []
    for i in range(self.num_hidden_layers):
      self.infopacker_layers.append(
          AggregatorBlock(
              base_vector_size=self.base_vector_size,
              num_info_heads=self.num_info_heads,
              size_per_head=self.size_per_head,
              infopacker_hidden_dropout_prob=self.infopacker_hidden_dropout_prob,
              initializer_range=self.initializer_range,
              infopacker_dropout_prob=self.infopacker_dropout_prob,
              name=f"layer_{i}"))
      
      self.info_extractor_builder = self.add_weight("info_extractor_builder",
                                                    shape=[self.size_per_head]*3,
                                                    initializer=get_initializer(self.initializer_range),
                                                    dtype=tf.float32,
                                                    trainable=True)

    super(Aggregator, self).build(input_shapes)

  def call(self, input_tensor, return_all_layers=False):
    all_layer_outputs = []
    output_tensor = input_tensor

    for layer in self.infopacker_layers:
      output_tensor = layer(output_tensor, self.info_extractor_builder)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_hidden_layers': self.num_hidden_layers,
        'base_vector_size': self.base_vector_size,
        'num_attention_heads': self.num_attention_heads,
        'size_per_head': self.size_per_head,
        'infopacker_hidden_dropout_prob': self.infopacker_hidden_dropout_prob,
        'initializer_range': self.initializer_range,
        'infopacker_dropout_prob': self.infopacker_dropout_prob
    })
    return config 