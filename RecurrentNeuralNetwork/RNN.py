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


class RNN:
    singleInputLen = 100
    embeddingDimension = 256
    rnnUnits = 1024
    batchSize = 64
    bufferSize = 10000
    epochsNum = 2

    def __init__(self, vocabSize, dataAsInt):
        tf.enable_eager_execution()
        self.model = None
        self.LSTM = None
        self.dataSet = None
        self.checkpoint_dir = ''
        self.vocabSize = vocabSize
        self.dataAsInt = dataAsInt
        self.optimizer = tf.train.AdamOptimizer()
        self.CreateDataSet()
        self.DefineRNNModel()

    def CreateDataSet(self):
        # Convert the text vector into a stream of character indices
        charDataset = tf.data.Dataset.from_tensor_slices(self.dataAsInt)

        # Convert this stream of character indices to sequences of the desired size
        sequences = charDataset.batch(RNN.singleInputLen + 1, drop_remainder=True)

        # Define the input and target texts for each sequence
        self.dataSet = sequences.map(SplitInputTarget)

        # Shuffle the data for the purpose of stochastic gradient descent
        # Pack it into batches which will be used during training
        self.dataSet = self.dataSet.shuffle(self.bufferSize).batch(RNN.batchSize, drop_remainder=True)

    def DefineRNNModel(self):
        self.LSTM = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')
        self.LSTM = functools.partial(self.LSTM,
                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform',
                                      stateful=True)
        self.BuildModel(RNN.batchSize)
        print('RNN Model summary:')
        print(self.model.summary())

    def BuildModel(self, batchSize):
        self.model = tf.keras.Sequential(
            [tf.keras.layers.Embedding(self.vocabSize, RNN.embeddingDimension, batch_input_shape=[batchSize, None]),
             self.LSTM(RNN.rnnUnits),
             tf.keras.layers.Dense(self.vocabSize)])

    def TrainRNNModel(self):
        self.checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        history = []
        plotter = util.PeriodicPlotter(sec=1, xlabel='Iterations', ylabel='Loss')
        for epoch in range(RNN.epochsNum):
            self.model.reset_states()
            for inp, target in self.dataSet:
                with tf.GradientTape() as tape:
                    predictions = self.model(inp)
                    loss = ComputeLoss(target, predictions)

                # Compute the gradients and try to minimize
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Update the progress bar!
                history.append(loss.numpy().mean())
                plotter.plot(history)

            # Update the model with the changed weights!
            self.model.save_weights(checkpoint_prefix.format(epoch=epoch))
        print(f'Final loss %: {loss.numpy().mean()}')

    def RebuildModelWithDifferentBatch(self, newBatch):
        self.BuildModel(newBatch)
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.build(tf.TensorShape([newBatch, None]))

    def PredictText(self, start_string, char2idx, idx2char, generation_length=1000):
        # Evaluation step (generating text using the learned RNN model)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        self.model.reset_states()
        text_generated = []

        for i in range(generation_length):
            predictions = self.model(input_eval)
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()
            # Pass the prediction along with the previous hidden state
            # as the next inputs to the model
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)


def ComputeLoss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def SplitInputTarget(chunk):
    inputText = chunk[:-1]
    targetText = chunk[1:]
    return inputText, targetText


def GenerateData():
    tempStr = ''
    for i in range(0, 10000):
        tempStr += 'kostya'
    with open('myData.txt', 'a') as fileObj:
        fileObj.write(tempStr)


def GetDataFromFile():
    with open('myData.txt', 'r') as fileObj:
        fileStr = fileObj.read()
        print(f'Length of text: {len(fileStr)} characters\n')
        print(f'First chars: {fileStr[:250]}\n')
        return fileStr


def CreateCharIntDB(fileStr):
    vocabulary = sorted(set(fileStr))
    char2idx = {u: i for i, u in enumerate(vocabulary)}
    idx2char = np.array(vocabulary)
    dataAsInt = np.array([char2idx[c] for c in fileStr])

    print(f'File has {len(vocabulary)} unique characters\n')
    return [vocabulary, char2idx, idx2char, dataAsInt]


def main():
    # GenerateData()
    fileStr = GetDataFromFile()
    [vocabulary, char2idx, idx2char, dataAsInt] = CreateCharIntDB(fileStr)

    rnnInstance = RNN(len(vocabulary), dataAsInt)
    rnnInstance.TrainRNNModel()
    rnnInstance.RebuildModelWithDifferentBatch(1)
    text = rnnInstance.PredictText('k', char2idx, idx2char)
    print(f'Predicted text:\n{text}')


if __name__ == "__main__": main()
