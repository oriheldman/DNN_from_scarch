import json
import matplotlib.pyplot as plt
from DNN import *
import time

# # # # # # # Research Code #  # # # # # # # # # #
def standarize_data(x, train_mean=None, train_std=None):
    """
    Normalize the pixel values of the input data
    :param x: The data to be normalized (array of pixel values from 0-255)
    :return: the normalized data
    """

    eps = 0.000001
    if train_mean is None:
        train_mean = x.mean(axis=1, keepdims=True)
        train_std = x.std(axis=1, keepdims=True) + eps

    x_normed = (x - train_mean) / train_std
    return x_normed, train_mean, train_std


def plot_loss_graph(train_loss, val_loss, saved_plot_path=None):
    epochs = list(range(len(train_loss)))

    plt.plot(epochs, train_loss, 'g', linestyle='solid', label='Train loss')
    plt.plot(epochs, val_loss, 'b', linestyle='solid', label='Validation loss')

    plt.title('Training and Validation Cost')
    plt.xlabel('Iterations (on 100)')
    plt.ylabel('Loss')
    plt.legend()

    if saved_plot_path:
        plt.savefig(saved_plot_path)

    plt.show()


experiment = 'Baseline'
# experiment = 'WithBN'
# experiment = 'WithBN_and_Dropout-0.1'
# experiment = 'WithDropout-0.1'

start = time.time()
print(experiment)


X_train, X_test, y_train, y_test = load_data()
learning_rate = 0.009
csize = 10
input_size = 28 * 28  # 784
num_iterations = 50000
batch_size = 128

# DNN architecture - input size, size of each layer and number of class to classify
layers_dims = np.array([input_size, 20, 7, 5, csize])

# Training the model
parameters, train_costs, val_costs, val_acc, iter, epoch = L_layer_model(X_train, y_train, layers_dims, learning_rate,
                                                                         num_iterations, batch_size=batch_size)

end = time.time()
print(end - start)

#  Predicting on test set using the DNN parameters after training
test_acc = Predict(X_test, y_test, parameters)


plot_loss_graph(train_costs, val_costs, saved_plot_path=experiment + '.png')
experiment_dict = {'Title': experiment, 'Epochs': epoch+1, 'n_iterations': iter+1, 'Batch_size': batch_size,
                   'val_acc': val_acc, 'test_acc': test_acc, 'Train_Time': int(end - start)}

with open(experiment + '.json', 'w') as fp:
    json.dump(experiment_dict, fp)
