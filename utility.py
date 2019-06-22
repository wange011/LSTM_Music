import os, random
import numpy as np

from data_formating import *

def loadPianoPieces():
    working_directory = os.getcwd()
    music_directory = working_directory + "/training_sets/piano/"

    midi_directories = ["albeniz"]
    #, "beeth", "borodin", "brahms", "burgm", "chopin", "debussy", "granados", "grieg", "haydn", "liszt", "mendelssohn", "mozart", "muss", "schubert", "schumann", "tschai"]
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

def getPieceSegment(pieces, num_time_steps):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0,len(piece_output)-num_time_steps,division_len)
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-num_time_steps,division_len, start)

    seg_out = piece_output[start:start+num_time_steps]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def getPieceBatch(pieces, batch_size, num_time_steps):
    i,o = zip(*[getPieceSegment(pieces, num_time_steps) for _ in range(batch_size)])
    return np.array(i), np.array(o)

def generateMIDI(piece, name):
    working_directory = os.getcwd()
    output_directory = working_directory + "/sample_outputs/"
    output_name = output_directory + name    
    
    noteStateMatrixToMidi(piece, output_name)            