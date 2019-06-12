import tensorflow as tf

import random
import numpy as np
import os

import data_formating

def generatePieces(model_name, time_block_outputs, X, hidden_state, y, outputs, output_parameters):
    
    steps_trained = output_parameters["steps_trained"]
    num_pieces = output_parameters["num_pieces"]
    timesteps = output_parameters["timesteps"]    
    
    # Finds the most recent saved model
    steps_trained = steps_trained - (steps_trained % 200)

    working_directory = os.getcwd()    
    saver = tf.train.Saver()    

    # Returns np array of shape (num_pieces, timesteps = 1, 78, 2)        
    pieces = randomTimestep(num_pieces)       
    
    with tf.Session() as sess:
        
        saver.restore(sess, working_directory + "/saved_models/" + model_name + "_" + str(steps_trained) + "_iterations.ckpt")                 
            
        for step in range(1, timesteps + 1):
            
            # Use data_formating
            current_timestep_input = generateInputFromPreviousTimestep(pieces)
            
            # Run the time block by one tick, run entire recurrent sequence of note-axis layers, pass that final output into the next time step layer
            # (batch_size, timesteps, state_size)
            # where batch_size is each individual note for every song in the song batch (78 * song_batch_size)            
            time_block_outputs_run = sess.run([time_block_outputs], feed_dict={X: current_timestep_input})
            
            hidden_state_size = tf.size(time_block_outputs_run)[2]            
            
            time_block_outputs_run = tf.reshape(time_block_outputs_run, [78, num_pieces, 1, hidden_state_size])            
            
            labels = np.zeros((num_pieces, 1, 2))

            hidden_state_run =  tf.slice(time_block_outputs_run, [0, 0, 0, 0], [1, num_pieces, 1, hidden_state_size])           
            
            for i in range(78):
                # Pass in by adding a note each time, sample from last output (batch_size: num_pieces in feed_dict) 
                # Append that to labels
                hidden_state_run = tf.transpose(hidden_state_run, perm=[1, 0, 2, 3])
                hidden_state_run = tf.reshape(hidden_state_run, [num_pieces, i + 1, hidden_state_size])          
            
                outputs = sess.run([outputs], feed_dict={hidden_state: hidden_state_run, y: labels})
                
                # Outputs in shape (song_batch_size, song_timesteps = 1, num_notes = 1, 2)
                # Samples from solely last input of the num_notes dimension
                note_step = sampleFromOutputs(outputs)
                # Returns np array of size (song_batch_size, 1, 2) 
                labels = np.concat((labels, note_step), axis=1)
                
                if i < 77:
                    hidden_state_run = tf.slice(time_block_outputs_run, [0, 0, 0, 0], [i + 2, num_pieces, 1, hidden_state_size])
            
            # Remove the first row of labels
            labels = labels[:, 1:, :]
            labels = np.reshape(labels, (num_pieces, 1, 78, 2))
            # Add labels as a timestep of pieces
            pieces = np.concat((pieces, labels), axis=1)
            
        return pieces
        

def generateInputFromPreviousTimestep(pieces):

    num_pieces = len(pieces)
    timestep_num = len(pieces[0]) - 1

    pieces = pieces[:, timestep_num, :, :]
    pieces = np.concat((pieces, np.zeros((num_pieces, 1, 78, 2))), axis=1)

    biaxial_inputs = data_formating.noteStateToBiaxialInput(pieces[0], timestep_num)[1, :, :53]
    biaxial_inputs = np.reshape(biaxial_inputs, [1, 1, 78, 53])    
    
    for i in range(1, num_pieces):    
        next_input = data_formating.noteStateToBiaxialInput(pieces[i], timestep_num)[1, :, :53]
        next_input = np.reshape(next_input, [1, 1, 78, 53])  
        biaxial_inputs = np.concat((biaxial_inputs, next_input), axis=0)        

    return biaxial_inputs    

# Returns np array of shape (batch_size, timesteps = 1, 78, 2)    
def randomTimestep(batch_size):
    
    timestep = np.zeros((batch_size, 1, 78, 2))        

    for i in range(batch_size):
        rand_note = random.randint(0, 79)
        timestep[i][0][rand_note][0] = 1
        timestep[i][0][rand_note][1] = 1

    return timestep

# Outputs in shape (song_batch_size, song_timesteps = 1, num_notes = 1, 2)
# Samples from solely last input of the num_notes dimension
# Returns np array of size (song_batch_size, 1, 2) 
# Accounts for dropout (.5 during training) by multiplying output by .5
def sampleFromOutputs(outputs):
    
    batch_size = outputs.get_shape().as_list()[0]
    
    sample = np.zeros((batch_size, 1, 2))
    
    for i in range(batch_size):
        play = random.uniform()
        
        if play <= outputs[i][0][0][0] * 0.5:
            sample[i][0][0] = 1
        
        articulate = random.uniform()
        
        if articulate <= outputs[i][0][0][1] * 0.5:
            sample[i][0][1] = 1

"""
def generatePiece(input_dim = 2 * 78):

    working_directory = os.getcwd()    
    saver = tf.train.Saver()    
    
    piece = []
    with tf.Session() as sess:
        
        print("Generating Seed")
        
        rand = random.randint(0, 78)   
        
        seed = np.zeros(input_dim)
        seed[rand] = 1
        seed[rand + 78] = 1              
        seed = np.reshape(seed, [1, 1, input_dim])    
    
        print("Generating Piece")    
        
        saver.restore(sess, working_directory + "/saved_models/model_50x0_iterations.ckpt")    
        
        for step in range(1, 100):
            
            outputs_run, = sess.run([outputs], feed_dict={X: seed})
            
            notes = np.array(outputs_run)[0, 0, :78]
            articulation = np.array(outputs_run)[0, 0, 78:]        
            
            sample = sampleTimestep(notes, articulation)        
            
            seed = outputs_run
    
            piece.append(sample)
    
    piece = np.array(piece)
    utility.generateMIDI(piece)

def sampleTimestep(notes, articulation):
    notes = tf.convert_to_tensor(notes)
    notes = tf.distributions.Bernoulli(logits=notes).sample().eval()

    for i in range(len(notes)):
        if notes[i] == 0:
            articulation[i] = 0
    
    articulation = tf.convert_to_tensor(articulation)
    articulation = tf.distributions.Bernoulli(logits=articulation).sample().eval()
    
    return np.vstack((notes, articulation)).T    
"""    