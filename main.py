import tensorflow as tf

import os 
import numpy as np
import utility
import model
import training
#import generate_music

working_directory = os.getcwd()

# MIDI files are first converted to Note State

"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""

# Gather the training pieces from the specified directories
# Saved with dimensions (timesteps, notes_played, articulation)
training_set, testing_set = utility.loadPianoPieces()
print(training_set['alb_esp2'][3])   

tf.reset_default_graph()


X = tf.placeholder("float", [None, None, 78, 12])
time_hidden_layer_size = [300, 300]

batch_size = tf.shape(X)[0]
timesteps = tf.shape(X)[1]

time_block_outputs = model.BiaxialTimeBlock(X, time_hidden_layer_size)


time_block_hidden = tf.reshape(time_block_outputs, [batch_size, 78, timesteps, time_block_outputs.get_shape().as_list()[2]])
time_block_hidden = tf.transpose(time_block_hidden, perm=[2, 0, 1, 3])    
time_block_hidden = tf.reshape(time_block_hidden, [batch_size * timesteps, 78, time_block_hidden.get_shape().as_list()[3]])    
        

hidden_state = tf.placeholder("float", [None, None, 300])
y = tf.placeholder("float", [None, None, None, 2])
note_hidden_layer_size = [100, 50]


generating_music = tf.Variable(initial_value=False, dtype="bool")


time_block_hidden = tf.cond(tf.equal(generating_music, True), true_fn=lambda: hidden_state, false_fn=lambda: time_block_hidden)

outputs = model.BiaxialNoteBlock(time_block_hidden, y, note_hidden_layer_size, batch_size, timesteps)

loss = model.BiaxialLoss(outputs, y)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


model_name = "BiaxialLSTM"
timesteps = 16
batch_size = 5
steps = 10000
display_step = 500

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