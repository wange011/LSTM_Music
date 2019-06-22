import tensorflow as tf

import random
import numpy as np
import os

import data_formating
import model
import utility

def generatePieces(model_name, time_block_outputs, X, hidden_state, generating_music, y, outputs, output_parameters):
    
    steps_trained = output_parameters["steps_trained"]
    num_pieces = output_parameters["num_pieces"]
    timesteps = output_parameters["timesteps"]
    display_step = output_parameters["display_step"]    
    
    # Finds the most recent saved model
    steps_trained = steps_trained - (steps_trained % display_step)

    working_directory = os.getcwd()    
    saver = tf.train.Saver()    

    # Returns np array of shape (num_pieces, timesteps = 1, 78, 2)        
    pieces = randomTimestep(num_pieces)       
    
    with tf.Session() as sess:
        
        saver.restore(sess, working_directory + "/saved_models/" + model_name + "_" + str(steps_trained) + "_iterations.ckpt")                 
        
        begin_generating = generating_music.assign(True)

        sess.run(begin_generating)        
        
        for step in range(1, timesteps + 1):
            
            # Use data_formating
            current_timestep_input = generateInputFromPreviousTimestep(pieces)
            
            # Run the time block by one tick, run entire recurrent sequence of note-axis layers, pass that final output into the next time step layer
            # (batch_size, timesteps, state_size)
            # where batch_size is each individual note for every song in the song batch (78 * song_batch_size)            
            time_block_outputs_run, = sess.run([time_block_outputs], feed_dict={X: current_timestep_input})           
            
            hidden_state_size = time_block_outputs_run.shape[2]            
            
            time_block_outputs_run = np.reshape(time_block_outputs_run, (78, num_pieces, 1, hidden_state_size))            
            
            labels = np.zeros((num_pieces, 1, 1, 2))

            hidden_state_run =  time_block_outputs_run[:1, :, :, :]           
            
            for i in range(78):
                # Pass in by adding a note each time, sample from last output (batch_size: num_pieces in feed_dict) 
                # Append that to labels
                hidden_state_run = np.transpose(hidden_state_run, (1, 0, 2, 3))
                hidden_state_run = np.reshape(hidden_state_run, (num_pieces, i + 1, hidden_state_size))          
            
                outputs_run, = sess.run([outputs], feed_dict={X: current_timestep_input, hidden_state: hidden_state_run, y: labels})
                
                # Outputs in shape (song_batch_size, song_timesteps = 1, num_notes = 1, 2)
                # Samples from solely last input of the num_notes dimension
                note_step = sampleFromOutputs(outputs_run)
                # Returns np array of size (song_batch_size, 1, 1, 2)
                labels = np.concatenate((labels, note_step), axis=2)
                
                if i < 77:
                    hidden_state_run = time_block_outputs_run[:i+2, :, :, :] 
            
            # Remove the first row of labels
            labels = labels[:, :, 1:, :]
            # Add labels as a timestep of pieces
            pieces = np.concatenate((pieces, labels), axis=1)
            
        return pieces
        

def generateInputFromPreviousTimestep(pieces):

    num_pieces = len(pieces)
    timestep_num = len(pieces[0]) - 1

    pieces = pieces[:, timestep_num:, :, :]

    pieces = np.concatenate((pieces, np.zeros((num_pieces, 1, 78, 2))), axis=1)

    biaxial_inputs = data_formating.noteStateToBiaxialInput(pieces[0], timestep_num)[1, :, :53]
    biaxial_inputs = np.reshape(biaxial_inputs, [1, 1, 78, 53])    
    
    for i in range(1, num_pieces):    
        next_input = data_formating.noteStateToBiaxialInput(pieces[i], timestep_num)[1, :, :53]
        next_input = np.reshape(next_input, [1, 1, 78, 53])  
        biaxial_inputs = np.concatenate((biaxial_inputs, next_input), axis=0)        

    return biaxial_inputs    

# Returns np array of shape (batch_size, timesteps = 1, 78, 2)    
def randomTimestep(batch_size):
    
    timestep = np.zeros((batch_size, 1, 78, 2))        

    for i in range(batch_size):
        rand_note = random.randint(0, 78)
        timestep[i][0][rand_note][0] = 1
        timestep[i][0][rand_note][1] = 1

    return timestep

# Outputs in shape (song_batch_size, song_timesteps = 1, num_notes = 1, 2)
# Samples from solely last input of the num_notes dimension
# Returns np array of size (song_batch_size, 1, 2) 
# Accounts for dropout (.5 during training) by multiplying output by .5
def sampleFromOutputs(outputs):

    batch_size = outputs.shape[0]
    
    for i in range(batch_size):
        outputs[i][0][0][0] = sigmoid(outputs[i][0][0][0])
        outputs[i][0][0][1] = sigmoid(outputs[i][0][0][1])
    
    sample = np.zeros((batch_size, 1, 1, 2))
    
    for i in range(batch_size):
        play = random.uniform(0, 1)
       
        """
        if play <= outputs[i][0][0][0]: #* 0.5:
            sample[i][0][0][0] = 1
        
            articulate = random.uniform(0, 1)
        
            if articulate <= outputs[i][0][0][1]: #* 0.5:
                sample[i][0][0][1] = 1

        """
        if 0.5 <= outputs[i][0][0][0]: #* 0.5:
            sample[i][0][0][0] = 1
        
            articulate = random.uniform(0, 1)
        
            if 0.5 <= outputs[i][0][0][1]: #* 0.5:
                sample[i][0][0][1] = 1
        
    
    return sample
    
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm    
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

if __name__ == "__main__":
    
    
    tf.reset_default_graph()


    X = tf.placeholder("float", [None, None, 78, 53])
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
    optimizer = tf.train.AdadeltaOptimizer()
    train_op = optimizer.minimize(loss)


    model_name = "BiaxialLSTM"
    
    output_parameters = {"steps_trained": 50000, "num_pieces": 1, "timesteps": 100, "display_step": 500}
    pieces = generatePieces(model_name, time_block_outputs, X, hidden_state, generating_music, y, outputs, output_parameters)
    
    print(pieces)    
    
    for j in range(len(pieces)):
        utility.generateMIDI(pieces[j], model_name + "_" + str(50000) + "_iterations_test_" + str(j + 1))
    
