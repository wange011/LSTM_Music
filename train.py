import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

import os 

import utility

def train(loss, log_likelihood):
    
    working_directory = os.getcwd()
    # Initialize the variables for the computational graph
    init = tf.global_variables_initializer()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    timesteps = 16
    batch_size = 15
    training_steps = 5000
    
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
            
            if step % display_step == 0 or step == 1:
    
                # To restore model during training: saver.restore(sess, "/tmp/model.ckpt")
                
                # Saves the model            
                save_path = saver.save(sess, working_directory + "/saved_models/model_" + str(step) + "_iterations.ckpt")            
                
                # Calculate batch loss and accuracy
                print("Step " + str(step) + ", Loss= " + str(loss_run) + ", Log Likelihood= " + str(log_likelihood_run))
                
# define resumeTraining()