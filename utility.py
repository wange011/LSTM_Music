import os, random
import numpy as np
from data_formating import *

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
        print ("Loaded {}".format(name))
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
    
    