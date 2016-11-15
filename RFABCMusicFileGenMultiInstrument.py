import numpy as np
import ast
from sklearn.cross_validation import train_test_split
from sklearn import tree
np.set_printoptions(threshold=np.nan)
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import numpy as np

def TrainRandomForest(newsong_data, NotesCreated, minusindex,data, startingnote, newsong):
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:cols]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_y = np.ravel(train_y)
    rfclf.fit(train_x,train_y)
    newsong_data = newsong_data.reshape(1,-1)
    intnewnote = int(rfclf.predict(newsong_data))
    newnote = outputscale[int(rfclf.predict(newsong_data))]
    char = ''
    if len(newsong) > 5:
        if any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 2:
                minusindex = 3
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 1:
                minusindex = 2
            else:
                minusposition = (newsong[len(newsong) - minusindex].index('-'))
                newnotevar1 = newsong[len(newsong) - minusindex][0:minusposition]
                newnotevar2 = newsong[len(newsong) - minusindex][(minusposition+1):]
                newnote = newnotevar1 + newnotevar2
                if any(i  in newnote for i in addingties) > 0:
                    minusposition = newnote.index('-')
                    newnotevar1 = newnote[0:minusposition]
                    newnotevar2 = newnote[(minusposition+1):]
                    newnote = newnotevar1 + newnotevar2
                    print(newnote)

    if len(newsong) == NotesCreated-1 and any(i in newnote for i in addingties):
        minusposition = newnote.index('-')
        newnotevar1 = newnote[0:minusposition]
        newnotevar2 = newnote[(minusposition + 1):]
        newnote = newnotevar1 + newnotevar2

    if any(i in newnote for i in addingextranoteslist) and NotesCreated == startingnote + 1:
        NotesCreated = NotesCreated + 1
    newsong[len(newsong)] = newnote
    newsong_data = np.append(newsong_data,np.zeros((1,UniqueCharacter)))
    newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
    return newsong_data, NotesCreated, minusindex

def Preprocess():
    i = 0
    while i < len(dataname):
        lines = open(dataname[i]).readlines()
        if instrument in ['drums']:
            lines = lines[8:]
        s = ''
        for line in lines:
            if not line.startswith('%'):
                s += line
        s = s.replace('M: ', '')
        s = s.replace('Q: ', ' ')
        s = s.replace('K: ', ' ')
        s = s.replace('C maj', 'Cmaj ')
        s = s.replace('|', '| ')
        s = s.replace('\n', '')
        s = s.replace('\t', ' ')
        s = s.split(' ')
        if i == 0:
            abcdata = np.asarray(s)
            newabcdata = np.asarray(s)
        else:
            newabcdata = np.asarray(s)
        if abcdata.shape[0] > newabcdata.shape[0]:
            while abcdata.shape[0] > newabcdata.shape[0]:
                newabcdata = np.append([newabcdata], [newabcdata[numdescriptors:]])  # Tempo, Timing, Key
                newabcdata = newabcdata[0:abcdata.shape[0]]
        else:
            while abcdata.shape[0] < newabcdata.shape[0]:
                abcdata = np.append([abcdata], [abcdata[numdescriptors:]])  # Tempo, Timing, Key
                abcdata = abcdata[0:newabcdata.shape[0]]
        abcdata = np.append([abcdata], [newabcdata], axis=0)
        i = i + 1
    print('Num Notes: ', abcdata.shape[1])
    return abcdata

def CreateDictionary(outputscale,inputscale):
    i = 0
    if instrument in ['drums']:
        n = 3
    else:
        n = 0
    while i < abcdata.shape[0]:
        while n < LengthMusic + numdescriptors:
            note = abcdata[i,n]
            try:
                foo = inputscale[note]
            except KeyError:
                outputscale[len(outputscale)] = str(note)
                inputscale = {v: k for k, v in outputscale.items()}
            n = n + 1
        if instrument in ['drums']:
            n = 3
        else:
            n = 0
        i = i + 1
    print('Output Scale: ',outputscale)
    print('Length Output Scale: ',len(outputscale))
    return outputscale, inputscale

def BuildOneHot(input_data):
    if instrument in ['drums']:
        input_data = np.zeros(shape=(abcdata.shape[0], num_features + lendescriptors), dtype=np.int)
        i = 0
        n = 0
        while i < abcdata.shape[0]:
            input_data[i,0] = Tempo[i]
            input_data[i,inputtimemetric[Timing[i]]] = 1
            input_data[i,inputkey[Key[i]]+len(outputtimemetric)] = 1
            k = 0
            while n < LengthMusic:
                note = abcdata[i,(n+numdescriptors)]
                input_data[i,k+inputscale[note]+lendescriptors]=1
                n = n + 1
                k = k + UniqueCharacter
            n=0
            i = i + 1
        return input_data
    else:
        oldinput_data = input_data
        lenoldinput = oldinput_data.shape[1]
        newinput_zeros = np.zeros(shape=(abcdata.shape[0], num_features), dtype=np.int)
        input_data = np.concatenate((oldinput_data,newinput_zeros),axis=1)
        i = 0
        n = 0
        while i < abcdata.shape[0]:
            k=0
            while n < LengthMusic:
                note = abcdata[i,n]
                input_data[i, k + inputscale[note] + lenoldinput] = 1
                n = n + 1
                k = k + UniqueCharacter
            n=0
            i = i + 1
        return input_data

def CreateTrack(NotesCreated,newsong):
    minusindex = 1
    if instrument in ['drums']:
        songtempo, songtiming, songkey = abcdata[0, 1], abcdata[0, 0], abcdata[0, 2]
        note1, note2, note3 = outputscale[0], outputscale[1], outputscale[2]
        outputstring = note1 + ' ' + note2 + ' ' + note3
        newsong = {0: note1, 1: note2, 2: note3}
        songtiming = inputtimemetric[songtiming]
        songkey = inputkey[songkey]
        note1 = inputscale[note1]
        note2 = inputscale[note2]
        note3 = inputscale[note3]
        notes = {0: note1, 1: note2, 2: note3}
        startingnote = 3
        newsong_data = np.zeros((1, lendescriptors + UniqueCharacter * startingnote))
        newsong_data[0, 0] = songtempo
        newsong_data[0, songtiming] = 1
        newsong_data[0, len(outputtimemetric) + songkey] = 1
        newsong_data[0, lendescriptors + UniqueCharacter * 0 + notes[0]] = 1
        newsong_data[0, lendescriptors + UniqueCharacter * 1 + notes[1]] = 1
        newsong_data[0, lendescriptors + UniqueCharacter * 2 + notes[2]] = 1
        while startingnote < NotesCreated:
            y = np.zeros((abcdata.shape[0], 1))
            X = input_data[:, 0:lendescriptors + UniqueCharacter * startingnote]
            ynoteonehot = input_data[:, lendescriptors + UniqueCharacter * startingnote:lendescriptors + (startingnote * UniqueCharacter) + UniqueCharacter]
            n = 0
            i = 0
            while i < abcdata.shape[0]:
                while n < UniqueCharacter:
                    if ynoteonehot[i, n] == 1:
                        newvar = n / UniqueCharacter
                        note = inputscale[outputscale[round((newvar % 1) * UniqueCharacter)]]
                        y[i] = int(note)
                    n = n + 1
                n = 0
                i = i + 1
            data = np.append(X, y, axis=1)
            newsong_data, NotesCreated, minusindex = TrainRandomForest(newsong_data, NotesCreated, minusindex, data, startingnote, newsong)
            # newsong_data, NotesCreated = TrainSimpleNN(newsong_data, NotesCreated)
            startingnote = startingnote + 1
        return newsong_data, newsong, outputstring
    else:
        startingnote = 0
        while startingnote < NotesCreated:
            y = np.zeros((abcdata.shape[0], 1))
            X = input_data[:, 0:previnput_data.shape[1] + UniqueCharacter * startingnote]
            newsong_data = X[0,:]
            ynoteonehot = input_data[:,previnput_data.shape[1] + UniqueCharacter * startingnote:previnput_data.shape[1] + (startingnote * UniqueCharacter) + UniqueCharacter]
            n = 0
            i = 0
            while i < abcdata.shape[0]:
                while n < UniqueCharacter:
                    if ynoteonehot[i, n] == 1:
                        newvar = n / UniqueCharacter
                        note = inputscale[outputscale[round((newvar % 1) * UniqueCharacter)]]
                        y[i] = int(note)
                    n = n + 1
                n = 0
                i = i + 1
            data = np.append(X, y, axis=1)
            newsong_data, NotesCreated, minusindex = TrainRandomForest(newsong_data, NotesCreated, minusindex, data, startingnote, newsong)
            # newsong_data, NotesCreated = TrainSimpleNN(newsong_data, NotesCreated)
            startingnote = startingnote + 1
        return newsong_data, newsong

def OutputFile(outputstring):
    OutTempo = newsong_data[0]
    n = 1
    while n < len(outputtimemetric) + 1:
        if newsong_data[n] == 1:
            OutTiming = outputtimemetric[n]
        n = n + 1
    n = len(outputtimemetric)
    while n < len(outputtimemetric) + len(inputkey) + 1:
        if newsong_data[n] == 1:
            OutKey = outputkey[n - len(outputtimemetric)]
        n = n + 1
    i = 3
    while i < NotesCreated:
        outputstring = outputstring + ' ' + newsong[i]
        i = i + 1
    outputstring = outputstring.replace(' ] ', '')
    text_file = open("Test.txt", 'w')
    text_file.write("X: 1" + "\nT: Drums" + "\nM: " + str(OutTiming) + "\nQ: " + str(int(OutTempo)) + "\nK: " + str(OutKey) + "\n\n" + outputstring)
    i = NotesCreated
    outputstring = ''
    while i < NotesCreated*2:
        outputstring = outputstring + ' ' + newsong[i]
        i = i + 1
    outputstring = outputstring.replace(' ] ', '')
    text_file.write("\n\nX: 2" + "\nT: Theorbo" + "\n\n" + outputstring)
    i = NotesCreated*2
    outputstring = ''
    while i < NotesCreated*3:
        outputstring = outputstring + ' ' + newsong[i]
        i = i + 1
    outputstring = outputstring.replace(' ] ', '')
    text_file.write("\n\nX: 3" + "\nT: Lute of Ages" + "\n\n" + outputstring)
    i = NotesCreated*3
    outputstring = ''
    while i < NotesCreated*4:
        outputstring = outputstring + ' ' + newsong[i]
        i = i + 1
    outputstring = outputstring.replace(' ] ', '')
    text_file.write("\n\nX: 4" + "\nT: Clarinet" + "\n\n" + outputstring)
    text_file.close()

rfclf = tree.DecisionTreeClassifier()

addingextranoteslist = ('|','+mp+','+mf+','+f+','+ff+','+fff+','+ffff+','+p+','+pp+','+ppp+','+pppp+')
addingties = ('-')
outputinstrument = {1:'Drums',2:'Theorbo',3:'LuteofAges',4:'Clarinet',5:'Flute'}
outputtimemetric = {1:'4/4',2:'3/4',3:'1/4',4:'2/4',5:'5/8',6:'6/8'}
outputkey = {1:'Cmaj'}
outputsetting = {1:'Field',2:'Town',3:'Desert',4:'Ice',5:'Battle',6:'Boss'}
outputmood = {1:'Happy',2:'Sad'}

inputinstrument = {v: k for k, v in outputinstrument.items()}
inputtimemetric = {v: k for k, v in outputtimemetric.items()}
inputkey = {v: k for k, v in outputkey.items()}
inputsetting = {v: k for k, v in outputsetting.items()}
inputmood = {v: k for k, v in outputmood.items()}

lendescriptors = 1 + len(outputtimemetric) + len(outputkey)
numdescriptors = 3

game2, song2 = 'MysticalNinja', 'ShikokuIsland'

game1, song1 = 'FinalFantasy6', 'Terra'

instrument = 'drums'

#dataname = {0:'OneInstrument/ABCFav/DKC2StickerBrushSymphony.abc',1:'OneInstrument/ABCFav/DKC1ForestFrenzy.abc'}
dataname = {0:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument),1:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument)}

NotesCreated = 150

abcdata = Preprocess()
Timing, Tempo, Key = abcdata[:,0], abcdata[:,1], abcdata[:,2]
LengthMusic = abcdata.shape[1]-numdescriptors                                            #3 = Tempo, TimeMetric, Key
outputscale, inputscale = {}, {}
outputscale, inputscale = CreateDictionary(outputscale,inputscale)
UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic
input_data = 0
input_data = BuildOneHot(input_data)
previnput_data = input_data
newsong = {}
newsong_data,newsong,outputstring = CreateTrack(NotesCreated,newsong)

instrument = 'bass'
numdescriptors = 0
lendescriptors = 0
dataname = {0:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument),1:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument)}
abcdata = Preprocess()
LengthMusic = abcdata.shape[1]
outputscale, inputscale = CreateDictionary(outputscale, inputscale)
UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic
input_data = BuildOneHot(input_data)
print(input_data.shape)
newsong_data,newsong = CreateTrack(NotesCreated,newsong)
previnput_data = input_data

instrument = 'lute'
numdescriptors = 0
lendescriptors = 0
dataname = {0:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument),1:'MultiInstrument/{}/{}/{}.abc'.format(game2,song2,instrument)}
abcdata = Preprocess()
LengthMusic = abcdata.shape[1]
outputscale, inputscale = CreateDictionary(outputscale, inputscale)
UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic
input_data = BuildOneHot(input_data)
newsong_data,newsong = CreateTrack(NotesCreated,newsong)
previnput_data = input_data

instrument = 'clarinet'
numdescriptors = 0
lendescriptors = 0
dataname = {0:'MultiInstrument/{}/{}/{}.abc'.format(game2,song2,instrument),1:'MultiInstrument/{}/{}/{}.abc'.format(game1,song1,instrument)}
abcdata = Preprocess()
LengthMusic = abcdata.shape[1]
outputscale, inputscale = CreateDictionary(outputscale, inputscale)
UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic
input_data = BuildOneHot(input_data)
newsong_data,newsong = CreateTrack(NotesCreated,newsong)

OutputFile(outputstring)