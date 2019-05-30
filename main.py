import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os 
import random

import utility
import model

# Parse throught the training MIDI files
working_directory = os.getcwd()
music_directory = working_directory + "/training/piano/"

midi_directories = ["albeniz"]
#, "beeth", "borodin", "brahms", "burgm", "chopin", "debussy", "granados", "grieg", "haydn", "liszt", "mendelssohn", "mozart", "muss", "schubert", "schumann", "tschai"]
max_time_steps = 256 # only files at least this many 16th note steps are saved
num_validation_pieces = 1

# MIDI files are converted to Note State
"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""
# Gather the training pieces from the specified directories
training = {}
for i in range(len(midi_directories)):
    folder = music_directory + midi_directories[i]
    training = {**training, **utility.loadPieces(folder, max_time_steps)}


# Set aside a random set of pieces for testing
testing = {}
for i in range(num_validation_pieces):
    index = random.choice(list(training.keys()))
    testing[index] = training.pop(index)

# Use utility.getPieceBatch(training_set, batch_size, time_steps)[1] to get batch

# Input dimensions
timesteps = 16
input_dim = 78 * 2

# Hyperparameters
learning_rate = 0.001
training_steps = 10000
batch_size = 15
num_hidden = [100, 100, 100, 100]
num_layers = 2

# Here None is the batch_size
# Define the Graph Input
X = tf.placeholder("float", [None, timesteps, input_dim])

# !!!!!!!Define the weight matrices, use LSTMStateTuple to pass in the weights as LSTM States
# tf.nn.dynamic_rnn outputs output and cell_state
# Use MultiRNNCell to instantiate multilayer LSTM (pass in cell_list)


# 2 consectutive 2 layer LSTMs
# Decide how to separate the 2 parameters: note played and articulation
h = model.TimewiseLSTM(X, num_hidden[:num_layers - 1])
outputs = model.NotewiseLSTM(h, num_hidden[num_layers:])

# Define loss and optimizer
loss = model.LossFunction(outputs, X, batch_size, timesteps, input_dim)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

# Training
display_step = 100

init = tf.global_variables_intitializer()

with tf.session() as sess:
    
    sess.run(init)

    for step in range(1, training_steps + 1):
        
        # Get batch of shape
        batch = utility.getPieceBatch(training, batch_size, timesteps)[1]
        
        # !!!!Reshape batch
        
        
        loss = sess.run(train_op, feed_dict={X: batch})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            print("Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss))
