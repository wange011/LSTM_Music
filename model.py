import tensorflow as tf
#from tensorflow.contrib.rnn import BasicLSTMCell
#from tensorflow.contrib.rnn import DropoutWrapper


def BiaxialTimeBlock(X, time_hidden_layer_size):
    
    # Input Dimensions: (song_batch_size, song_timesteps, 78, input_dim)
    song_batch_size = tf.shape(X)[0] 
    song_timesteps = tf.shape(X)[1] 
    input_dim = X.get_shape().as_list()[3] 

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
    time_block_inputs = tf.transpose(X, perm=[0, 2, 1, 3])
    time_block_inputs = tf.reshape(time_block_inputs, [song_batch_size * 78, song_timesteps, input_dim])
    
    #time_block_inputs = tf.keras.Input((song_batch_size * 78, song_timesteps, input_dim), tensor=time_block_inputs)    
    
    timewise_lstm_stack = []

    for i in range(2):
        
        timewise_lstm_stack.append(tf.keras.layers.LSTMCell(time_hidden_layer_size[i]))
        
        #timewise_lstm_stack.append(DropoutWrapper(BasicLSTMCell(time_hidden_layer_size[i], forget_bias=1.0), output_keep_prob=.5))

    
    #timewise_lstm_stack = tf.keras.layers.StackedRNNCells(timewise_lstm_stack)    
    #timewise_lstm_stack = tf.contrib.rnn.MultiRNNCell(timewise_lstm_stack)
    
    time_block_outputs = tf.keras.layers.RNN(timewise_lstm_stack, return_sequences=True)(time_block_inputs)      
    #time_block_outputs, time_block_state = tf.nn.dynamic_rnn(cell=timewise_lstm_stack, inputs=time_block_inputs, dtype=tf.float32)

    return time_block_outputs

def BiaxialNoteBlock(hidden_state, y, note_hidden_layer_size, song_batch_size, song_timesteps):

    num_notes = tf.shape(hidden_state)[1]
    #hidden_state_size = tf.shape(labels)[3]
    
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
    and timesteps corresponds to each individual note at a particular timestep in the song (78 during training) 

    Outputs will be of dimension: (batch_size, timesteps, state_size)
    """
    
    # Append Previous Note Played and Previous Note Articulated
    y = tf.slice(y, [0, 0, 0, 0], [song_batch_size, song_timesteps, num_notes - 1, 2])

    zeros = tf.zeros([song_batch_size, song_timesteps, 1, 2], dtype="float")
        
    labels = tf.concat([zeros, y], 2) 

    labels = tf.reshape(labels, [song_batch_size * song_timesteps, num_notes, 2])    
    
    # Append time_block_output with previous_note_played and previous_note_articulated
    note_block_inputs = tf.concat([hidden_state, labels], 2)
    
    #note_block_inputs = tf.keras.Input((song_batch_size * song_timesteps, num_notes, hidden_state_size), tensor=note_block_inputs)    
    
    notewise_lstm_stack = []

    for i in range(2):
        
        notewise_lstm_stack.append(tf.keras.layers.LSTMCell(note_hidden_layer_size[i]))
        #notewise_lstm_stack.append(DropoutWrapper(BasicLSTMCell(note_hidden_layer_size[i], forget_bias=1.0), output_keep_prob=.5))
      
    #notewise_lstm_stack = tf.keras.layers.StackedRNNCells(notewise_lstm_stack)    
    #notewise_lstm_stack = tf.contrib.rnn.MultiRNNCell(notewise_lstm_stack)
    
    note_block_outputs = tf.keras.layers.RNN(notewise_lstm_stack, return_sequences=True)(note_block_inputs)    
    #note_block_outputs, noteblock_state = tf.nn.dynamic_rnn(cell=notewise_lstm_stack, inputs=note_block_inputs, dtype=tf.float32)    
    
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
    
    outputs = tf.keras.layers.Dense(2)(note_block_outputs)

    return outputs

def BiaxialLoss(outputs, labels):
    """
    Computes sigmoid cross entropy given logits (network outputs)
    
    Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive

    We have 2 independent classes for each note: (note_played, note_articulation), so we use sigmoid cross entropy    
    """

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs)

    loss = tf.reduce_mean(loss)    
    
    return loss

"""
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
"""    