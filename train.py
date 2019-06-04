import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

import os 
import random

import utility
import model


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Parse throught the training MIDI files
working_directory = os.getcwd()
music_directory = working_directory + "/training/piano/"

midi_directories = ["albeniz", "beeth", "borodin", "brahms", "burgm", "chopin", "debussy", "granados", "grieg", "haydn", "liszt", "mendelssohn", "mozart", "muss", "schubert", "schumann", "tschai"]
max_time_steps = 256 # only files at least this many 16th note steps are saved
num_testing_pieces = 1

# MIDI files are converted to Note State

"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""

# Gather the training pieces from the specified directories
# Saved with dimensions (timesteps, notes_played, articulation)
training = {}
for i in range(len(midi_directories)):
    folder = music_directory + midi_directories[i]
    training = {**training, **utility.loadPieces(folder, max_time_steps)}


# Set aside a random set of pieces for testing
testing = {}
for i in range(num_testing_pieces):
    index = random.choice(list(training.keys()))
    testing[index] = training.pop(index)

# Use utility.getPieceBatch(training_set, batch_size, time_steps)[1] to get batch

tf.reset_default_graph()

# Input dimensions
input_dim = 78 * 2

# Hyperparameters
learning_rate = 0.001
training_steps = 5000
num_hidden = [200, input_dim]
num_layers = 2

# Here None is the batch_size, timesteps
# Define the Graph Input
X = tf.placeholder("float", [None, None, input_dim])

# !!!!!!!Define the weight matrices, use LSTMStateTuple to pass in the weights as LSTM States
# tf.nn.dynamic_rnn outputs output and cell_state
# Use MultiRNNCell to instantiate multilayer LSTM (pass in cell_list)


# 2 consectutive 2 layer LSTMs
# Uses the Biaxial model
outputs = model.TimewiseLSTM(X, num_hidden[:num_layers])
#outputs = model.NotewiseLSTM(h, num_hidden[num_layers:])

# Define loss and optimizer
# Returns the cross entropy loss based upon the next note to be played
loss, log_likelihood = model.LossFunction(outputs, X, input_dim)
# Using the Adam optimizer for now
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

# Training
display_step = 500
losses = []

# Initialize the variables for the computational graph
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

timesteps = 16
batch_size = 15


with tf.Session() as sess:
    
    print("Starting Training")
    sess.run(init)

    for step in range(1, training_steps + 1):
        
        
        # Get batch of shape [batch_size, timesteps, num_notes, 2]
        # Corresponds with values for (batch, timestep, note_played, articulation)
        batch = utility.getPieceBatch(training, batch_size, timesteps)[1]
        
        # Reshape to match input dimensions
        batch = np.reshape(batch, [batch_size, timesteps, input_dim])
        
        loss_run, log_likelihood_run, _ = sess.run([loss, log_likelihood, train_op], feed_dict={X: batch})
        losses.append(loss_run)        
        
        if step % display_step == 0 or step == 1:

            # To restore model during training: saver.restore(sess, "/tmp/model.ckpt")
            
            # Saves the model            
            save_path = saver.save(sess, working_directory + "/saved_models/model_" + str(step) + "_iterations.ckpt")            
            
            # Calculate batch loss and accuracy
            print("Step " + str(step) + ", Loss= " + str(loss_run))


# Generate new music
piece = []
with tf.Session() as sess:
    
    print("Generating Seed")
    
    rand = random.randint(0, 78)   
    
    seed = np.zeros(input_dim)
    seed[rand] = 1
    seed[rand + 78] = 1              
    seed = np.reshape(seed, [1, 1, input_dim])    

    print("Generating Piece")    
    
    saver.restore(sess, working_directory + "/saved_models/model_60_iterations.ckpt")    
    
    for step in range(1, 100):
        
        outputs_run, = sess.run([outputs], feed_dict={X: seed})
        
        notes = np.array(outputs_run)[0, 0, :78]
        articulation = np.array(outputs_run)[0, 0, 78:]        
        
        sample = utility.getSampleTimestep(notes, articulation)        
        
        seed = outputs_run

        piece.append(sample)

piece = np.array(piece)
utility.generateMIDI(piece)
            
