import tensorflow as tf

import random
import numpy as np
import os

import utility

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