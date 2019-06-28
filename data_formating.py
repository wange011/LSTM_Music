import midi, numpy as np
import itertools

lowerBound = 24
upperBound = 102

def midiToNoteStateMatrix(midifile):

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound-lowerBound
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

def noteStateMatrixToMidi(statematrix, name="example"):
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

lowerBound = 24
upperBound = 102

def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]
        
        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]
        
        return part_position + part_pitchclass + [0]*66 + [1] 
    return [noteSentinel(note) for note in range(upperBound-lowerBound)]

def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d

def buildContext(state):
    context = [0]*12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lowerBound) % 12
            context[pitchclass] += 1
    return context
    
def buildBeat(time):
    return [2*x-1 for x in [time%2, (time//2)%2, (time//4)%2, (time//8)%2]]

def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable((getOrDefault(state, note+i, [0,0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]

def noteStateSingleToInputForm(state,time):
    beat = buildBeat(time)
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat) for note in range(len(state))]

def noteStateMatrixToInputForm(statematrix):
    # NOTE: May have to transpose this or transform it in some way to make Theano like it
    #[startSentinel()] + 
    inputform = [ noteStateSingleToInputForm(state,time) for time,state in enumerate(statematrix) ]
    return inputform    

"""
Returns the Note State Matrix with the following dimensions:

Position [Size 1]:
Pitchclass [12]:
Previous Vicinity [24]:
Previous Context [12]:
Beat [4]:

Labels:
Played
Articulated

Output should have shape (timesteps, num_notes, 1 + 12 + 24 + 12 + 4 + 2 = 55)
"""    
def noteStateToBiaxialInput(statematrix, timestep_num = 0):
    
    biaxial_input = []   
    
    beat = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
            [0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0],
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]       
    
    for timestep in range(len(statematrix)):

        timestep_matrix = []        
        
        for note in range(len(statematrix[0])):
             
            note_matrix = []
            
            note_matrix.append(note)
            note_matrix.extend([1 if x == note % 12 else 0 for x in range(12)])
            
            prev_vicinity = [0 for x in range(24)]
            if (timestep != 0):
                
                for i in range(len(statematrix[timestep - 1])):                
                    
                    if statematrix[timestep - 1][i][0] == 1:
                        prev_vicinity[2*((note - i) % 12)] = 1
                    
                    if statematrix[timestep - 1][i][1] == 1:
                        prev_vicinity[2*((note - i) % 12) + 1] = 1
                
            note_matrix.extend(prev_vicinity)
            
            prev_context = [0 for x in range(12)]
            if (timestep != 0):
                
                for i in range(len(statematrix[timestep - 1])):                
                    
                    if statematrix[timestep - 1][i][0] == 1:
                        prev_context[(note - i) % 12] = prev_context[(note - i) % 12] + 1
                         
            note_matrix.extend(prev_context)
            note_matrix.extend(beat[(timestep + timestep_num) % 16])
            
            note_matrix.append(statematrix[timestep][note][0])            
            note_matrix.append(statematrix[timestep][note][0])            
            
            timestep_matrix.append(note_matrix)

        biaxial_input.append(timestep_matrix)        
        
    return np.array(biaxial_input)            