import tensorflow as tf
from tensorflow.contrib import rnn


def BiaxialTimeBlock(inputs, time_hidden_layer_size):
    
    # Input Dimensions: (song_batch_size, song_timesteps, 78, input_dim)
    song_batch_size = len(inputs) 
    song_timesteps = len(inputs[0])
    input_dim = len(inputs[0][0][0])

    """
    Time Block:
    2 layer LSTM with recurrent connections along the time axis
    
    Input Features:
    Position [Size 1]:
    Pitchclass [12]:
    Previous Vicinity [24]:
    Previous Context [12]:
    Beat [4]:

    tf.keras.layers.RNN accepts inputs with shape: (batch_size, timesteps, ...)

    Reshape inputs to be: (batch_size, timesteps, 1 + 12 + 24 + 12 + 4 = 53)
    where batch_size is each individual note for every song in the song batch (78 * song_batch_size)

    Outputs will be of dimension: (batch_size, timesteps, state_size)

    """ 
    # Reshaping the inputs
    time_block_inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])   
    time_block_inputs = tf.reshape(time_block_inputs, [song_batch_size * 78, song_timesteps, input_dim])
        
    # time_hidden_layer_size = [300, 300]

    timewise_lstm_stack = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(n, dropout=.5) for n in time_hidden_layer_size])    
    
    time_block = tf.keras.layers.RNN(timewise_lstm_stack, return_sequences=True)    
    
    time_block_outputs = time_block(time_block_inputs)    
    
    return time_block_outputs

def BiaxialNoteBlock(hidden_state, labels, note_hidden_layer_size):

    song_batch_size = len(labels) 
    song_timesteps = len(labels[0])
    num_notes = len(labels[0][0])  
    
    """
    Note Block:
    2 layer LSTM with recurrent connections along the note axis
    
    Input Features:
    Time Block Output [h]:
    Previous Note Played [1]:
    Previous Note Articulated [1]:

    tf.keras.layers.RNN accepts inputs with shape: (batch_size, timesteps, ...)

    Reshape inputs to be: (batch_size, timesteps, h + 2)
    where batch_size is each timestep for each song in the song batch    
    and timesteps corresponds to each individual note at a particular timestep in the song (78) 

    Outputs will be of dimension: (batch_size, timesteps, state_size)
    """
    
    note_block_inputs = tf.concat([hidden_state, labels], 2)
          
    notewise_lstm_stack = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(n, dropout=.5) for n in note_hidden_layer_size])    
    
    note_block = tf.keras.layers.RNN(notewise_lstm_stack, return_sequences=True)    
    
    note_block_outputs = note_block(note_block_inputs)   
    
    """
    Final FeedForward Layer to determine play and articulation probabilities    

    tf.keras.layers.Dense accepts inputs with shape: (batch_size, ..., input_dim)
    where input_dim will be equal to the last hidden layer size of the Note Block
    
    Final output will consist of 2 values for each note:
    (note_played, note_articulated)
    
    Outputs will be of dimension: (song_batch_size, song_timesteps, num_notes, 2)
    """    
     # Convert the outputs of the note block to the input shape of the Dense layer
    note_block_outputs = tf.reshape(note_block_outputs, [song_batch_size, song_timesteps, num_notes, note_hidden_layer_size[-1]])
    
    dense_layer = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)
    
    outputs = dense_layer(note_block_outputs)

    return outputs

def BiaxialLoss(outputs, labels):
    """
    Computes sigmoid cross entropy given logits (network outputs)
    
    Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive

    We have 2 independent classes: (note_played, note_articulation), so we use sigmoid cross entropy    
    """

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs)

    return loss

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