import tensorflow as tf
from tensorflow.contrib import rnn

# Try dropout
def TimewiseLSTM(X, num_hidden):
    
    cell_list = []
    
    for i in range(num_hidden):
        
        lstm_cell = rnn.BasicLSTMCell(num_hidden[i], forget_bias=1.0)
        cell_list.append(lstm_cell)
    
    multi_lstm_cell = rnn.MultiRNNCell(cell_list)    
    
    outputs, states = rnn.dynamic_rnn(multi_lstm_cell, X, dtype=tf.float32)
    
def NotewiseLSTM(h, num_hidden):
    
    cell_list = []
    
    for i in range(num_hidden):
        
        lstm_cell = rnn.BasicLSTMCell(num_hidden[i], forget_bias=1.0)
        cell_list.append(lstm_cell)
    
    multi_lstm_cell = rnn.MultiRNNCell(cell_list)    
    
    outputs, states = rnn.dynamic_rnn(multi_lstm_cell, h, dtype=tf.float32)
    
def LossFunction(outputs, X, batch_size, timesteps, input_dim):
    
    # Remove the first slice of X and last slice of outputs
    outputs_sliced = tf.slice(outputs, [0, 0, 0], [batch_size, timesteps - 1, input_dim])
    X_sliced = tf.slice(X, [0, 1, 0], [batch_size, timesteps - 1, input_dim])    
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs_sliced, labels=X_sliced)        
    
    loss = tf.reduce_mean(cross_entropy)
    log_likelihood = -loss * input_dim    
    
    return loss, log_likelihood