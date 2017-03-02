from __future__ import absolute_import
import numpy as np
import os
import itertools
import codecs
import sys
from collections import Counter
import re
import argparse

np.random.seed(0)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

# output model directory
# need a different model directory per configuration
model_dir = 'models/' # 'models10' refer to the 10th model in my experiment

# Training parameters
embeddingsDim = 200
batchSize = 128
hiddenSize = 200
posSize = 100
epochs = 1000
maxLenSentence = 0

parser = argparse.ArgumentParser()
parser.add_argument('-train',action='store',dest='trainFileName',help='Training File Path')
parser.add_argument('-embeddings', action='store', dest='embeddingsDim', help='Embeddings dimension',default=200)
parser.add_argument('-output',action='store',dest='outputName',help='Output file name (including models)')
parser.add_argument('-batch',action='store',dest='batchSize', help='batch size',default=32)
parser.add_argument('-vocab',action='store',dest='vocab',help='vocab size',default=10000)
parser.add_argument('-hidden',action='store',dest='hiddenSize',help='hidden layer size',default=200)
parser.add_argument('-maxSequenceLength',action='store',dest='maxSequenceLength',help='hidden layer size',default=700)
parser.add_argument('-verbose',action='store',dest='verbose',help='verbose',default=1)
parser.add_argument('-test',action='store',dest='testFileName',help='Testing File Path')
parser.add_argument('-tune',action='store',dest='tuneFileName',help='Tuning File Path')

parameters = parser.parse_args()

embeddingsDim = int(parameters.embeddingsDim)
batchSize = int(parameters.batchSize)
hiddenSize = int(parameters.hiddenSize)
outputName = parameters.outputName
trainFileName = parameters.trainFileName
vocabSize = int(parameters.vocab)
verb = int(parameters.verbose)
testFileName = parameters.testFileName
tuneFileName = parameters.tuneFileName
maxSequenceLength = int(parameters.maxSequenceLength)
outFile = codecs.open(outputName + ".out",'w',encoding='latin-1')

print("embeddingsDim: " + str(embeddingsDim))
print("batchSize: " + str(batchSize))
print("hiddenSize: " + str(hiddenSize))
print("outputName: " + outputName)
print("trainFileName: " + trainFileName)
print("tuneFileName: " + tuneFileName)
print("testFileName: " + testFileName)
print("Vocab size: " + str(vocabSize))
print("maxSequenceLength: " + str(maxSequenceLength))


outFile.write("embeddingsDim: " + str(embeddingsDim) + "\n")
outFile.write("batchSize: " + str(batchSize) + "\n")
outFile.write("hiddenSize: " + str(hiddenSize) + "\n")
outFile.write("outputName: " + outputName + "\n")
outFile.write("trainFileName: " + trainFileName + "\n")
outFile.write("tuneFileName: " + tuneFileName + "\n")
outFile.write("testFileName: " + testFileName + "\n")
outFile.write("Vocab size: " + str(vocabSize) + "\n")
outFile.write("maxSequenceLength: " + str(maxSequenceLength) + "\n")
outFile.write("\n-------------------------------------------------\n\n")

from keras.preprocessing import sequence
from keras.models import Model, Sequential, model_from_json
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.utils.np_utils import to_categorical

vocab = ['<PAD>','<UNK>']

labels_2dig = []
labels_4dig = []

trainingInstances = []
trainingLabels_2dig = []
trainingLabels_4dig = []
tuningInstances = []
tuningLabels_2dig = []
tuningLabels_4dig = []
testingInstances = []
testingLabels_2dig = []
testingLabels_4dig = []

trainFile = codecs.open(trainFileName,encoding='latin-1')
trainSentences = trainFile.read().lower().strip().split("\n")
testFile = codecs.open(testFileName,encoding='latin-1')
testSentences = testFile.read().lower().strip().split("\n")
tuneFile = codecs.open(tuneFileName,encoding='latin-1')
tuneSentences = tuneFile.read().lower().strip().split("\n")

tuneFile.close()
trainFile.close()
testFile.close()

tempVocab = []
print("Reading vocab")
for instance in trainSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	tempVocab.extend(sentence)
	labels_2dig.append(instance.strip().split(",")[0][:2])
	labels_4dig.append(instance.strip().split(",")[0][:])

for instance in tuneSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	tempVocab.extend(sentence)
	labels_2dig.append(instance.strip().split(",")[0][:2])
	labels_4dig.append(instance.strip().split(",")[0][:])

for instance in testSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	tempVocab.extend(sentence)
	labels_2dig.append(instance.strip().split(",")[0][:2])
	labels_4dig.append(instance.strip().split(",")[0][:])

vocabCounter = Counter(tempVocab).most_common()
for i in vocabCounter:
	if len(vocab) < vocabSize:
		vocab.append(i[0])

vocabDict = dict()
for i,value in enumerate(vocab):
	vocabDict[vocab[i]] = i

vocab = vocabDict
labels_2dig = list(set(labels_2dig))
labels_4dig = list(set(labels_4dig))

print("Reading training")
for instance in trainSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	labs = instance.strip().split(",")[0]
	for i,word in enumerate(sentence):
		if word not in vocab:
			word = "<UNK>"
		sentence[i] = vocab[word]
	trainingInstances.append(sentence)
	trainingLabels_2dig.append(labels_2dig.index(labs[:2]))
	trainingLabels_4dig.append(labels_4dig.index(labs[:]))
	if len(sentence) > maxLenSentence:
		maxLenSentence = len(sentence)

print("Reading tuning")
for instance in tuneSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	labs = instance.strip().split(",")[0]
	for i,word in enumerate(sentence):
		if word not in vocab:
			word = "<UNK>"
		sentence[i] = vocab[word]
	tuningInstances.append(sentence)
	tuningLabels_2dig.append(labels_2dig.index(labs[:2]))
	tuningLabels_4dig.append(labels_4dig.index(labs[:]))

print("Reading testing")
for instance in testSentences:
	sentence = instance.strip().split(",")[1].split(" ")
	labs = instance.strip().split(",")[0]
	for i,word in enumerate(sentence):
		if word not in vocab:
			word = "<UNK>"
		sentence[i] = vocab[word]
	testingInstances.append(sentence)
	testingLabels_2dig.append(labels_2dig.index(labs[:2]))
	testingLabels_4dig.append(labels_4dig.index(labs[:]))

if maxLenSentence > maxSequenceLength:
	maxLenSentence = maxSequenceLength

# for testing and debugging
'''
trainingInstances = trainingInstances[0:10]
trainingLabels = trainingLabels[0:10]
tuningInstances = tuningInstances[0:10]
tuningLabels = tuningLabels[0:10]
testingInstances = testingInstances[0:10]
testingLabels = testingLabels[0:10]
'''

trainingLabels_2dig = to_categorical(trainingLabels_2dig,nb_classes=len(labels_2dig))
trainingLabels_4dig = to_categorical(trainingLabels_4dig,nb_classes=len(labels_4dig))
testingLabels_2dig = to_categorical(testingLabels_2dig,nb_classes=len(labels_2dig))
testingLabels_4dig = to_categorical(testingLabels_4dig,nb_classes=len(labels_4dig))
tuningLabels_2dig = to_categorical(tuningLabels_2dig,nb_classes=len(labels_2dig))
tuningLabels_4dig = to_categorical(tuningLabels_4dig,nb_classes=len(labels_4dig))

vocabFile = codecs.open('vocab.txt','w',encoding='latin-1')
vocabFile.write(str(maxLenSentence) + "\n")
vocabFile.write("\n".join(vocab))
vocabFile.close()

labelsFile_2dig = codecs.open('labels.txt','w',encoding='latin-1')
labelsFile_2dig.write("\n".join(labels_2dig))
labelsFile_2dig.close()

labelsFile_4dig = codecs.open('labels.txt','w',encoding='latin-1')
labelsFile_4dig.write("\n".join(labels_4dig))
labelsFile_4dig.close()

trainingInstances = sequence.pad_sequences(trainingInstances,maxlen=maxLenSentence,value=0.)
tuningInstances = sequence.pad_sequences(tuningInstances,maxlen=maxLenSentence,value=0.)
testingInstances = sequence.pad_sequences(testingInstances,maxlen=maxLenSentence,value=0.)

# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel

filter_sizes = (3)
dropout_prob = (0.2,0.2,0.2)
num_filters = 100

graph_in = Input(shape=(maxLenSentence, embeddingsDim))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='same',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    #flatten = Flatten()(pool)
    flatten = LSTM(100)(pool)
    convs.append(flatten)
    
if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)

# main sequential model

main_input = Input(shape = (maxLenSentence,), dtype = 'int32', name = 'main_input')
x = Embedding(len(vocab), embeddingsDim, input_length=maxLenSentence)(main_input)
x = Dropout(dropout_prob[0])(x)
y = graph(x)
x = Dense(hiddenSize)(y)
x = Dropout(dropout_prob[1])(x)
x = Activation('relu')(x)
#x = Dense(len(labels_2dig))(x)
#out1 = Activation('softmax')(x)

#x = Merge(mode = 'concat')([y,out1])
x = Dense(hiddenSize)(x)
x = Dropout(dropout_prob[2])(x)
x = Activation('relu')(x)
x = Dense(len(labels_4dig))(x)
out2 = Activation('softmax')(x)

model = Model(input = main_input, output = out2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

# serialize model to JSON
model_json = model.to_json()
fname = model_dir + 'model.json'
with open(fname, "w") as json_file:
    json_file.write(model_json)

bestScore = 0
for e in range(epochs):
	print("Epoch: " + str(e))
	outFile.write("Epoch: " + str(e) + "\n")
	model.fit(trainingInstances,trainingLabels_4dig,batch_size=batchSize,nb_epoch=1,verbose=verb,class_weight='auto')

	print("Predicting tuning...")
	
	predictions = model.predict(tuningInstances,batch_size=batchSize,verbose=verb)
	
	outLabels = []
	
	for k in range(1):
		for i,row in enumerate(predictions):
			row = row.tolist()
			labs = []
			for j,score in enumerate(row):
				if float(score) >= 0.5:
					#if k == 0: labs.append(labels_2dig[j])
					labs.append(labels_4dig[j])
			labs = ",".join(labs)
			outLabels.append(tuneSentences[i] + "\t" + labs)
		
		fo = codecs.open('temp/output.' + outputName + '.tune.' + str(e) + '_' + str(k) + '.txt','w',encoding='latin-1')
		fo.write("\n".join(outLabels))
		fo.close()
		
	'''
	result = os.popen('python evaluate.py temp/output.' + outputName + '.tune.' + str(e) + '.txt').read().strip()
	print(result)
	outFile.write("Tuning results: " + str(result) + "\n")
	'''

	scores = model.evaluate(tuningInstances,[tuningLabels_2dig,tuningLabels_4dig])
	
	outFile.write("Tuning...\n")
	for i in range(len(model.metrics_names)):
		outFile.write(model.metrics_names[i] + "\t" + str(scores[i]*100))
		print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
		
	# x x 1 2 3 4 1 2 3 4
	# 0 1 2 3 4 5 6 7 8 9
	
	if bestScore <= scores[-1]:
		bestScore = scores[-1]
		fname = model_dir + 'best.model.h5'
		model.save_weights(fname,overwrite=True)
	
	'''
	print("Predicting testing...")
	
	predictions = model.predict(testingInstances,batch_size=batchSize,verbose=verb)
	
	outLabels = [[] for k in range(2)]
	for k in range(2):
		for i,row in enumerate(predictions[k]):
			row = row.tolist()
			labs = []
			for j,score in enumerate(row):
				if float(score) >= 0.5:
					if k == 0: labs.append(labels_2dig[j])
					else: labs.append(labels_4dig[j])
			labs = ",".join(labs)
			outLabels[k].append(testSentences[i] + "\t" + labs)
		
		fo = codecs.open('temp/output.' + outputName + '.test.' + str(e) + '_' + str(k) + '.txt','w',encoding='latin-1')
		fo.write("\n".join(outLabels[k]))
		fo.close()
	
	result = os.popen('python evaluate.py temp/output.' + outputName + '.test.' + str(e) + '.txt').read().strip()
	print(result)
	outFile.write("Testing results: " + str(result) + "\n")
	
	scores = model.evaluate(testingInstances,[testingLabels_2dig,testingLabels_4dig])
	outFile.write("Testing...\n")
	for i in range(len(model.metrics_names)):
		outFile.write(model.metrics_names[i] + "\t" + str(scores[i]*100))
		print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
	'''
outFile.close()
