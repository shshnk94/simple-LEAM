import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

def preprocess_sequence(text, opt):

   return pad_sequences(text, 
                        maxlen=opt.max_length,
                        padding="post")

def get_similar_words(label_embeddings, word_embeddings, opt):
    
    # Get the unnormalized similarity matrix G.
    sim = np.dot(label_embeddings, word_embeddings.T)
    
    # Sort the indices as per decreasing order of similarity scores
    indices = np.argsort(sim, axis=1,)
    for i in range(indices.shape[0]):
        indices[i] = indices[i][::-1]

    # Get most 'num_words' similar words.
    return [[opt.idx_to_word[indices[label, index]] 
           for index in range(opt.num_words)]
           for label in range(indices.shape[0])]

def plot_losses(training_loss, validation_loss):

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    ax0.plot(training_loss)
    ax0.set_ylabel("Training Loss")

    ax1.plot(validation_loss)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation Loss")
    
    plt.show()
