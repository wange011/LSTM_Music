import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

import os 

import utility
import model

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
training, testing = utility.loadPianoPieces()

# Use utility.getPieceBatch(training_set, batch_size, time_steps)[1] to get batch

tf.reset_default_graph()

# Input dimensions
input_dim = 78 * 2

# Hyperparameters
learning_rate = 0.001
training_steps = 500
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
train_op = optimizer.minimize(log_likelihood)

# Training
display_step = 50
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
            print("Step " + str(step) + ", Loss= " + str(loss_run) + ", Log Likelihood= " + str(log_likelihood_run))
            

# Load
# Model
# Loss
# Train
# Generate
