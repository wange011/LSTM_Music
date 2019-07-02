import tensorflow as tf

import os 
import utility
import model
import training

working_directory = os.getcwd()

# MIDI files are first converted to a noteStateMatrix

"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector (MIDI values are truncated between 24 and 102)
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""

# Gather the training pieces from the specified directories
# Converts the pieces from noteStateMatrix to the biaxialInputFormat
training_set, testing_set = utility.loadPianoPieces()   

tf.reset_default_graph()

# Inputs
X = tf.placeholder("float", [None, None, 78, 53])
time_hidden_layer_size = [300, 300]

batch_size = tf.shape(X)[0]
timesteps = tf.shape(X)[1]

# First run recurrent connections along the song timesteps, used to capture rhythm as well as phrasing
# Each of the 78 notes are passed through as inputs in the shape: (song_batch_size * 78, timesteps, input_dim) 
# Outputs are in the shape: (song_batch_size * 78, timesteps, state_size)
time_block_outputs = model.BiaxialTimeBlock(X, time_hidden_layer_size)

# Reshape time_block_outputs to be passed into the next LSTM block
# Need to be in the shape: (song_batch_size * timesteps, 78, state_size)
time_block_hidden = tf.reshape(time_block_outputs, [batch_size, 78, timesteps, time_block_outputs.get_shape().as_list()[2]])
time_block_hidden = tf.transpose(time_block_hidden, perm=[2, 0, 1, 3])    
time_block_hidden = tf.reshape(time_block_hidden, [batch_size * timesteps, 78, time_block_hidden.get_shape().as_list()[3]])    
        
# Input used during output generation (can have a variable number of notes)
hidden_state = tf.placeholder("float", [None, None, 300])
# Labels
y = tf.placeholder("float", [None, None, None, 2])
note_hidden_layer_size = [100, 50]

# Switch between output generation and training
# During training, time_block_outputs is passed to the next LSTM block
# During output generation, hidden_state is passed to the next LSTM block
generating_music = tf.Variable(initial_value=False, dtype="bool")
time_block_hidden = tf.cond(tf.equal(generating_music, True), true_fn=lambda: hidden_state, false_fn=lambda: time_block_hidden)

# Run recurrent connects along the note dimension, used to capture chords
# Each of the song timesteps are passed through as inputs in the shape: (song_batch_size * song_timesteps, num_notes, hidden_dim)
# hidden_dim will contain: the corresponding time_block_output for each note, whether the previous note was played, and whether the previous note was articulated 
outputs = model.BiaxialNoteBlock(time_block_hidden, y, note_hidden_layer_size, batch_size, timesteps)

# Define the loss function and optimizer
# Using sigmoid_cross_entropy for the multiclass classification problem
loss = model.BiaxialLoss(outputs, y)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Training parameters
model_name = "BiaxialLSTM"
timesteps = 128
batch_size = 5
steps = 50000
display_step = 1000

# Training the model
training_parameters = {"timesteps": timesteps, "batch_size": batch_size, "training_steps": steps, "display_step": display_step}

training.train(model_name, training_set, time_block_outputs, X, hidden_state, generating_music, y, outputs, loss, train_op, training_parameters)

"""
# Generating output samples
for i in range(1, int(steps / (2 * display_step) + 1)):

    steps_trained = i * 2 * display_step   

    print("Generating Pieces for " + model_name + "_" + str(steps_trained) + "_iterations")
    
    output_parameters = {"steps_trained": steps_trained, "num_pieces": 1, "timesteps": 50, "display_step": display_step}
    pieces = generate_music.generatePieces(model_name, time_block_outputs, X, hidden_state, generating_music, y, outputs, output_parameters)
    
    for j in range(len(pieces)):
        utility.generateMIDI(pieces[j], model_name + "_" + str(steps_trained) + "_iterations_" + str(j + 1))


output_parameters = {"steps_trained": 50000, "num_pieces": 1, "timesteps": 100, "display_step": 500}
pieces = generate_music.generatePieces(model_name, time_block_outputs, X, hidden_state, generating_music, y, outputs, output_parameters)
        
for j in range(len(pieces)):
    utility.generateMIDI(pieces[j], model_name + "_" + str(50000) + "_iterations_" + str(j + 1))
"""    
