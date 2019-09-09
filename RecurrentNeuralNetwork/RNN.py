# -*- coding: utf-8 -*-
"""
Example of Recurrent Neural Network (RNN) in action

@author: KocT9H
@Email:  koct9h@gmail.com

Based on MIT course:
© MIT 6.S191: Introduction to Deep Learning
introtodeeplearning.com

"""

import tensorflow as tf 
import numpy as np
import os
import functools
import util


def GenerateData():
    tempStr = ''
    for i in range(0,10000):
        tempStr += 'kostya'
    with open('myData.txt', 'a') as fileObj:
        fileObj.write(tempStr)

    
def GetDataFromFile():
    with open('myData.txt', 'r') as fileObj:
        fileStr = fileObj.read()
        print(f'Length of text: {len(fileStr)} characters\n')
        print(f'First chars: {fileStr[:250]}\n')
        return fileStr


def CreateCharIntDB(fileStr) :
    vocabulary = sorted(set(fileStr))
    char2idx = {u:i for i, u in enumerate(vocabulary)}
    idx2char = np.array(vocabulary)
    dataAsInt = np.array([char2idx[c] for c in fileStr])
    
    print(f'File has {len(vocabulary)} unique characters\n')
    return [vocabulary, char2idx, idx2char, dataAsInt]
    

def CreateTrainingExamples(dataLen, dataAsInt, singleInputLen):
    examplesPerEpoch = dataLen // singleInputLen
    
    # Convert the text vector into a stream of character indices
    charDataset = tf.data.Dataset.from_tensor_slices(dataAsInt)
    
    # Convert this stream of character indices to sequences of the desired size
    sequences = charDataset.batch(singleInputLen+1, drop_remainder=True)
    
    # Define the input and target texts for each sequence
    dataSet = sequences.map(SplitInputTarget)
    return [dataSet, examplesPerEpoch]


def SplitInputTarget(chunk):
    inputText = chunk[:-1]
    targetText = chunk[1:]
    return inputText, targetText


def CreateTrainingBatches(dataSet, examplesPerEpoch, BATCH_SIZE, BUFFER_SIZE):
    stepsPerEpoch = examplesPerEpoch // BATCH_SIZE
    # Shufle the data for the purpose of stochastic gradient descent
    # Pack it into batches which will be used during training
    dataSet = dataSet.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return [dataSet, stepsPerEpoch]
   
    
def DefineRNNModel(vocabSize, embeddingDimension, rnnUnits, batchSize):
    LSTM = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')
    LSTM = functools.partial(LSTM, 
                             return_sequences=True, 
                             recurrent_initializer='glorot_uniform',
                             stateful=True)
    model = BuildModel(LSTM,
                       vocabSize, 
                       embeddingDimension, 
                       rnnUnits, 
                       batchSize)
    print('RNN Model summary:')
    print(model.summary())
    return [model, LSTM]


def BuildModel(LSTM, vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocabSize, embeddingDim, 
                                      batch_input_shape=[batchSize, None]),
                                      LSTM(rnnUnits),
                                      tf.keras.layers.Dense(vocabSize)])
    return model
    

def TrainRNNModel(model, dataSet, EPOCHS_NUM):
    optimizer = tf.train.AdamOptimizer()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    history = []
    plotter = util.PeriodicPlotter(sec=1, xlabel='Iterations', ylabel='Loss')
    for epoch in range(EPOCHS_NUM):
        # Initialize the hidden state at the start of every epoch; initially is None
        hidden = model.reset_states()
        
        # Enumerate the dataset for use in training
        for inp, target in dataSet:
            with tf.GradientTape() as tape:
                predictions = model(inp)
                loss = ComputeLoss(target, predictions)

            # Compute the gradients and try to minimize
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Update the progress bar!
            history.append(loss.numpy().mean())
            plotter.plot(history)
        
        # Update the model with the changed weights!
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print(f'Final loss %: {loss.numpy().mean()}')
    return [model, checkpoint_dir]


def ComputeLoss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def RebuildModelWithDifferentBatch(LSTM, vocabSize, embeddingDim, rnnUnits, batchSize, checkpoint_dir):
    model = BuildModel(LSTM, vocabSize, embeddingDim, rnnUnits, batchSize)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    return model
    
    
def PredictText(model, start_string, char2idx, idx2char, generation_length=1000):
    # Evaluation step (generating text using the learned RNN model)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for i in range(generation_length):
        predictions = model(input_eval)
      
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
      
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
        # Pass the prediction along with the previous hidden state
        # as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)
      
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def main():
    singleInputLen = 100
    embeddingDimension = 256
    rnnUnits = 1024
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EPOCHS_NUM = 2
    
    #GenerateData()
    
    fileStr = GetDataFromFile()
    
    [vocabulary, char2idx, idx2char, dataAsInt] = CreateCharIntDB(fileStr)
    
    [dataSet, examplesPerEpoch] = CreateTrainingExamples(len(fileStr), dataAsInt, singleInputLen)
    
    [dataSet, stepsPerEpoc] = CreateTrainingBatches(dataSet, examplesPerEpoch, BATCH_SIZE, BUFFER_SIZE)
    
    [model, LSTM] = DefineRNNModel(len(vocabulary), embeddingDimension, rnnUnits, BATCH_SIZE)
    
    [model, checkpoint_dir] = TrainRNNModel(model, dataSet, EPOCHS_NUM)
    
    model = RebuildModelWithDifferentBatch(LSTM, 
                                           len(vocabulary), 
                                           embeddingDimension, 
                                           rnnUnits, 
                                           1, 
                                           checkpoint_dir)
    
    text = PredictText(model, 'k', char2idx, idx2char)
    print(f'Predicted text:\n{text}')
    
if __name__ == "__main__": main()