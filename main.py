import tensorflow as tf

import os 

import utility
import model
import training
import generate_music

working_directory = os.getcwd()

# MIDI files are converted to Note State

"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""

# Gather the training pieces from the specified directories
# Saved with dimensions (timesteps, notes_played, articulation)
training_set, testing_set = utility.loadPianoPieces()


tf.reset_default_graph()


X = tf.placeholder("float", [None, None, None])
time_hidden_layer_size = tf.placeholder("int", [2])

time_block_outputs = model.BiaxialTimeBlock(X, time_hidden_layer_size)


hidden_state = tf.placeholder("float", [None, None, None])
y = tf.placeholder("float", [None, None, None])
note_hidden_layer_size = tf.placeholder("int", [2])

outputs = model.BiaxialNoteBlock(hidden_state, y, note_hidden_layer_size)

loss = model.BiaxialLoss(outputs, y)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


model_name = "BiaxialLSTM"

# Training the model
training_parameters = {"timesteps": 16, "batch_size": 15, "training_steps": 2000, "display_step": 200}
model_parameters = {"time_hidden_layer_size" : [300, 300], "note_hidden_layer_size": [100, 50]}

training.train(model_name, training_set, X, time_hidden_layer_size, time_block_outputs, hidden_state, y, note_hidden_layer_size, loss, training_parameters, model_parameters)


# Generating output samples
for i in range(1, training_parameters["training_steps"] / (2 * training_parameters["display_step"]) + 1):

    steps_trained = i * 2 * training_parameters["display_step"]    
    
    output_parameters = {"steps_trained": steps_trained, "num_pieces": 5, "timesteps": 100}
    pieces = generate_music.generatePieces(model_name, time_block_outputs, X, hidden_state, y, outputs, output_parameters)
    
    for j in range(len(pieces)):
        utility.generateMIDI(pieces[j], model_name + "_" + steps_trained + "_iterations_" + str(j + 1))
