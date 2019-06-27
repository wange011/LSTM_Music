import os, random
import numpy as np

from data_formating import *

def loadPianoPieces():
    working_directory = os.getcwd()
    music_directory = working_directory + "/training_sets/piano/"

    midi_directories = ["albeniz", "beeth", "borodin", "brahms", "burgm", "chopin", "debussy", "granados", "grieg", "haydn", "liszt", "mendelssohn", "mozart", "muss", "schubert", "schumann", "tschai"]
    max_time_steps = 256 # only files at least this many 16th note steps are saved
    num_testing_pieces = 0
    
    training = {}
    for i in range(len(midi_directories)):
        folder = music_directory + midi_directories[i]
        training = {**training, **loadPieces(folder, max_time_steps)}


    # Set aside a random set of pieces for testing
    testing = {}
    for i in range(num_testing_pieces):
        index = random.choice(list(training.keys()))
        testing[index] = training.pop(index)
    
    for key in training.keys():
        training[key] = noteStateToBiaxialInput(training[key])

    for key in testing.keys():
        testing[key] = noteStateToBiaxialInput(testing[key])    
    
    return training, testing

def loadPieces(dirpath, max_time_steps):
    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]

        try:
            outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        except:
            print('Skip bad file = ', name)
            outMatrix=[]
            
        if len(outMatrix) < max_time_steps:
            continue

        pieces[name] = outMatrix
        # print ("Loaded {}".format(name))
    return pieces

division_len = 16

def generateBatches(training_set, batch_size, timesteps):

    batches = []    
    batch = []
    
    for key in training_set.keys():

        piece = training_set[key]        
        
        current_timestep = timesteps
        prev_timestep = 0        
        
        while current_timestep <= piece.shape[0]:
            
            batch.append(piece[prev_timestep:current_timestep, :, :])

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
            
            prev_timestep = current_timestep
            current_timestep += timesteps
        
    if len(batch) != 0:
        
        while len(batch) < batch_size:
            
            batch_num = random.randint(0, len(batches) - 1)
            sample_num = random.randint(0, batch_size - 1)
            batch.append(batches[batch_num][sample_num])            
            
        batches.append(batch)

    return np.array(batches)

def shuffleBatches(training_set):
    np.random.shuffle(training_set)
    return training_set

def generateMIDI(piece, name):
    working_directory = os.getcwd()
    output_directory = working_directory + "/sample_outputs/"
    output_name = output_directory + name    
    
    noteStateMatrixToMidi(piece, output_name)            