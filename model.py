import tensorflow as tf
from tensorflow.contrib import rnn

def TimewiseLSTM(X, num_hidden):
    
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
    outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
    
def NotewiseLSTM(h, num_hidden):
    
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
    outputs, states = rnn.static_rnn(lstm_cell, h, dtype=tf.float32)