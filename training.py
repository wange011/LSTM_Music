import tensorflow as tf
import numpy as np

import os 

import utility
import generate_music

def train(model_name, training_set, scales, time_block_outputs, X, hidden_state, generating_music, y, outputs, loss, train_op, notewise_train_op, training_parameters):
    
    working_directory = os.getcwd()
    
    timesteps = training_parameters["timesteps"]
    batch_size = training_parameters["batch_size"]
    training_steps = training_parameters["training_steps"]
    display_step = training_parameters["display_step"]    
        
    # Initialize the variables for the computational graph
    init = tf.global_variables_initializer()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    file = open("training_progress.txt","w")    
    
    with tf.Session() as sess:
        
        file.write("Starting Training")
        sess.run(init)
        
        # Generate mini-batches
        # Get batch of shape [batch_size, timesteps, 78, 55]
        training_set = utility.generateBatches(training_set, batch_size, timesteps)

        num_batches = training_set.shape[0]

        scales = utility.generateBatches(scales, batch_size, 1)

        num_scale_batches = scales.shape[0]

        for step in range(1, training_steps + 1):
            
            # Shuffle the training set between each epoch
            if step % num_batches == 1 and step != 1:
                training_set = utility.shuffleBatches(training_set)
            
            batch = training_set[(step - 1) % num_batches]
            
            inputs = batch[:, :, :, :53]
            labels = batch[:, :, :, 53:]

            # Evaluate the computational graph
            loss_run, outputs_run, _, = sess.run([loss, outputs, train_op], feed_dict={X: inputs, hidden_state: np.zeros((batch_size * timesteps, 78, 300)), y: labels})                
            
            # Each display_step iterations, save the model and generate outputs
            if step % display_step == 0:
                
                # Saves the model            
                saver.save(sess, working_directory + "/saved_models/" + model_name + "_" + str(step) + "_iterations.ckpt")            
                
                # Generates outputs
                output_parameters = {"steps_trained": step, "num_pieces": 2, "timesteps": 150, "display_step": display_step}
                pieces = generate_music.generatePieces(model_name, time_block_outputs, X, hidden_state, generating_music, y, outputs, output_parameters)
        
                for j in range(len(pieces)):
                    utility.generateMIDI(pieces[j], model_name + "_" + str(step) + "_iterations_" + str(j + 1))                
                    
                
                for i in range(timesteps):
                    for j in range(78):
                        outputs_run[0][i][j][0] = generate_music.sigmoid(outputs_run[0][i][j][0])
                        outputs_run[0][i][j][1] = generate_music.sigmoid(outputs_run[0][i][j][1])
                
                print(labels[0][3])
                # file.write("Labels:")
                # np.savetxt("training_progress.txt", labels[0][3], newline="")
                print(outputs_run[0][3])
                # file.write("Outputs:")
                # np.savetxt("training_progress.txt", outputs_run[0][3], newline="")
                
                # Calculate batch loss and accuracy
                print("Step " + str(step) + ", Loss= " + str(loss_run))
                file.write("Step " + str(step) + ", Loss= " + str(loss_run) + "\n")
            
            # Pretrain/readjust the network using scale chords
            # Done after the model is saved so that the saved model is not biased
            if step % 1000 == 0:

                print("Readjusting Model")                
                
                for notewise_step in range(1, 101):
                    
                    if notewise_step % num_batches == 1 and step != 1:
                        scales = utility.shuffleBatches(scales)
                    
                    batch = scales[(notewise_step - 1) % num_scale_batches]
            
                    inputs = batch[:, :, :, :53]
                    labels = batch[:, :, :, 53:]                   
                    
                    sess.run([notewise_train_op], feed_dict={X: inputs, hidden_state: np.zeros((batch_size * timesteps, 78, 300)), y: labels})
                    

"""                
def resumeTraining(model_name, training_set, steps_trained, time_block_outputs, loss, X, hidden_state, y, timesteps, batch_size, remaining_training_steps):

    # Finds the most recent saved model
    steps_trained = steps_trained - (steps_trained % 200)

    working_directory = os.getcwd()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()    
    
    with tf.Session() as sess:
        
        print("Resuming Training At Step " + str(steps_trained))
        saver.restore(sess, working_directory + "/saved_models/" + model_name + "_" + str(steps_trained) + "_iterations.ckpt")                 

        display_step = 200    
    
        for step in range(1, remaining_training_steps + 1):
            
            
            # Get batch of shape [batch_size, timesteps, 78, 55]
            # Corresponds with values for (batch, timestep, note_played, articulation)
            batch = utility.getPieceBatch(training_set, batch_size, timesteps)[1]
            
            inputs = batch[:][:][:][:53]
            labels = batch[:][:][:][53:]
            
            time_block_outputs_run = sess.run([time_block_outputs], feed_dict={X: inputs})

            time_block_outputs_run = tf.reshape(time_block_outputs_run, [batch_size, 78, timesteps, time_block_outputs_run.get_shape().as_list()[2]])
            time_block_outputs_run = tf.transpose(time_block_outputs_run, perm=[2, 0, 1, 3])    
            time_block_outputs_run = tf.reshape(time_block_outputs_run, [timesteps * batch_size, 78, time_block_outputs_run.get_shape().as_list()[2]])    
            
            # Append Previous Note Played and Previous Note Articulated
            labels = tf.slice(labels, [0, 0, 0, 0], [batch_size, timesteps, 77, 2])
            zeros = tf.zeros([batch_size, timesteps, 1, 2])
        
            labels = tf.concat([zeros, labels], 2)    
            
            labels = tf.reshape(labels, [timesteps * batch_size, 78, 2])
            
            
            note_block_inputs = tf.concat([time_block_outputs, labels], 2)            
            
            loss_run = sess.run([loss], feed_dict={hidden_state: note_block_inputs, y: labels})    
      
            
            if step % display_step == 0 or step == 1:
    
                # To restore model during training: saver.restore(sess, "/tmp/model.ckpt")
                
                # Saves the model            
                saver.save(sess, working_directory + "/saved_models/" + model_name + "_" + str(steps_trained + step) + "_iterations.ckpt")            
                
                # Calculate batch loss and accuracy
                print("Step " + str(step) + ", Loss= " + str(loss_run))                
"""                
