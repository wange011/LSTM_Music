import tensorflow as tf
from tensorflow.contrib import rnn


def BiaxialRNN():
    
    return outputs

# Try dropout
def TimewiseLSTM(X, num_hidden):
    
    cell_list = []
    
    for i in range(len(num_hidden)):
        
        lstm_cell = rnn.BasicLSTMCell(num_hidden[i], forget_bias=1.0)
        cell_list.append(lstm_cell)
    
    timewise_multi_lstm_cell = rnn.MultiRNNCell(cell_list)    
    
    outputs, states = tf.nn.dynamic_rnn(timewise_multi_lstm_cell, X, dtype=tf.float32)

    return outputs
    
def NotewiseLSTM(h, num_hidden):
    
    cell_list = []
    
    for i in range(len(num_hidden)):
        
        lstm_cell = rnn.BasicLSTMCell(num_hidden[i], forget_bias=1.0)
        cell_list.append(lstm_cell)
    
    notewise_multi_lstm_cell = rnn.MultiRNNCell(cell_list)    
    
    outputs, states = tf.nn.dynamic_rnn(notewise_multi_lstm_cell, h, dtype=tf.float32)
    
    return outputs
    
def LossFunction(outputs, X, input_dim):
    
    # Remove the first slice of X and last slice of outputs
    outputs_sliced = tf.slice(outputs, [0, 0, 0], [tf.shape(outputs)[0], tf.shape(outputs)[1] - 1, input_dim])
    X_sliced = tf.slice(X, [0, 1, 0], [tf.shape(outputs)[0], tf.shape(outputs)[1] - 1, input_dim])    
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs_sliced, labels=X_sliced)        
    
    loss = tf.reduce_mean(cross_entropy)
    log_likelihood = -loss * input_dim    
    
    return loss, log_likelihood