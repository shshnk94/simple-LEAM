from keras.layers import *
from keras.models import Model
from keras import optimizers
import keras.backend as K

import numpy as np
import pickle as pkl

from utils import *

class Options(object):
   
    def __init__(self):
        
        self.num_words = 6
        self.max_epochs = 20
        self.max_length = 100
        self.batch_size = 128
        self.lr = 0.01
        self.momentum = 0.0        
        self.class_number = 2
        self.class_names = ["good", "bad"]
        self.save_path = "../output/"
        
	
def get_label_embeddings(opt):

    label_embeddings = opt.glove_embeddings[[
                       opt.word_to_index[label]
                       for label in opt.class_names]]

    return Embedding(input_dim=label_embeddings.shape[0],
                     output_dim=label_embeddings.shape[1],
                     input_length=2,
                     weights=[label_embeddings],
                     trainable=True,
                     name="label_embedding_layer")

if __name__ == "__main__":
	
    opt = Options()
		
    # Read the glove embeddings from yelp_glove.p
    with open("../data/yelp_full_glove.p", "rb") as f:
         opt.glove_embeddings = pkl.load(f, encoding="latin1")
    
    # Read the data and the dictionaries
    with open("../data/yelp_sampled.pkl", "rb") as f:
        (train_text,
        val_text, 
        test_text, 
        train_label, 
        val_label,
        test_label, 
        opt.word_to_index, 
        opt.idx_to_word) = pkl.load(f, encoding='latin1')
    
    # Preprocess train, validation, and test data to have
    # max_length = 100 by padding zeros.
    train_text = preprocess_sequence(train_text, opt)
    val_text = preprocess_sequence(val_text, opt)
    test_text = preprocess_sequence(test_text, opt)

    # Get words and labels as inputs
    sequence = Input(shape=(opt.max_length,),
                     batch_shape=(None, opt.max_length),
                     name="sequence_input",
                     dtype="int32")
    
    labels = Input(shape=(opt.class_number,),
                  batch_shape=(None, opt.class_number),
                  name="label_input",
                  dtype="int32")

    # Get label and word embeddings for the above inputs	
    label_embeddings = get_label_embeddings(opt)(labels)

    word_embeddings = Embedding(input_dim=opt.glove_embeddings.shape[0],
                                output_dim=opt.glove_embeddings.shape[1],
                                input_length=opt.max_length,
                                weights=[opt.glove_embeddings],
                                trainable=False,
                                name="word_embedding_layer")(sequence)

    # Calculate the cosine similarity between the words and the
    # embedding. This results in a (opt.class_number x
    # opt.max_length) size matrix on which we apply the softmax.
    cosine_similarities = Dot(axes=2)([label_embeddings,
                                       word_embeddings])
	
    # Flatten the similarities before passing it to softmax.
    flattened_similarities = Flatten()(cosine_similarities)

    # Final softmax layer
    predictions = Dense(opt.class_number, 
                        activation="softmax")(flattened_similarities)

    model = Model(inputs=[sequence, labels], outputs=[predictions])

    # Print the model architecture
    print(model.summary())
	
	# Optimizer and model initialization
    sgd = optimizers.SGD(lr=opt.lr, momentum=opt.momentum)
    model.compile(optimizer=sgd,
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
   
    # FIXME: Constant input to labels each sample of the batch.
    # This is a hack. Still exploring a more elegant way to do
    # this.
    constant_label_input_train = np.tile(np.arange(opt.class_number),
                                         (train_text.shape[0], 1))
    constant_label_input_val = np.tile(np.arange(opt.class_number),
                                       (val_text.shape[0], 1))
                                  
    history = model.fit(x=[train_text,constant_label_input_train],
                        y=train_label,
                        validation_data=([val_text,
                                          constant_label_input_val], 
                                          val_label), 
                        epochs=opt.max_epochs, 
                        batch_size=opt.batch_size) 

    label_embedding_weights = model.get_layer(
                                    "label_embedding_layer").get_weights()
    word_embedding_weights = model.get_layer(
                                   "word_embedding_layer").get_weights()

 
    similar_words = get_similar_words(label_embedding_weights[0],
                                      word_embedding_weights[0],
                                      opt)

    # Write outputs to a ../output/similar_words.txt
    with open("../output/similar_words.txt", "w") as f:
        for index, label in enumerate(opt.class_names):

            f.write("Words similar to " + label + ": ")
            for word in similar_words[index]:
                f.write(word + " ")

            f.write("\n")
 
    plot_losses(history.history["loss"],
                history.history["val_loss"])
