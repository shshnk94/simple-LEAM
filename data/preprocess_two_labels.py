import pickle as pkl
import numpy as np

def create_data_samples(X, Y, num_samples):

    Y = np.array(Y).squeeze()

    all_indices = np.arange(len(X))
    bad_worst_indices = np.random.choice(
                        all_indices[(Y.argmax(axis=1) < 2).squeeze()], 
                        num_samples)
    good_best_indices = np.random.choice(
                        all_indices[(Y.argmax(axis=1) > 2).squeeze()], 
                        num_samples)
    
    X_sampled = [X[index] for index in bad_worst_indices]
    Y_sampled = np.repeat([1., 0.], num_samples, axis=0)
	
    X_sampled += [X[index] for index in good_best_indices]
    Y_sampled = np.vstack((Y_sampled,
                           np.repeat([0., 1.], num_samples, axis=0)))

    Y_sampled = np.expand_dims(Y_sampled.argmax(axis=0), axis=1)

    return X_sampled, Y_sampled
	

if __name__ == "__main__":

    with open("yelp_full.p", "rb") as handle:
        (train_text,
        val_text, 
        test_text, 
        train_label, 
        val_label,
        test_label, 
        idx_to_word, 
        word_to_index) = pkl.load(handle, encoding='latin1')


    X_train, Y_train = create_data_samples(train_text, train_label, 2500)
    X_val, Y_val = create_data_samples(val_text, val_label, 250)
    X_test, Y_test = create_data_samples(test_text, test_label, 250)

    with open("yelp_sampled.pkl", "wb") as handle:
         pkl.dump([X_train,
                   X_val, 
                   X_test,
                   Y_train,
                   Y_val, 
                   Y_test,
                   idx_to_word,
                   word_to_index], handle)
