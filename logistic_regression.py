import numpy as np
from sklearn.model_selection import train_test_split
import time

from preprocess import DataManager

# TODO: implement dropout

class Input:
  def __init__(self, input_dim):
    """
    Args:
      input_dim:          Input vector demension,
    """
    self.output_unit = input_dim
    self.output = None
    self.next_layer = None
  def __call__(self, input):
    self.output = input
    self.next_layer._feed_fwd()
  def _back_fwd(self, dError_dOut, learning_rate, momentum=0.0):
    # Do nothing
    pass
class Dense:
  def __init__(self, output_unit, activation_fn, bias_random_init=False, weight_random_init=False):
    """
    Args:
      activation_fn:      Activation function name
      bias_random_init:   Indicate whether bias should be initialized randomly,
      weight_random_init: Indicate whether weight should be initialized randomly
    """
    self.bias_random_init = bias_random_init
    self.weight_random_init = weight_random_init
    assert isinstance(activation_fn, str)
    self.activation_fn = {
        'sigmoid': lambda x: 1 / 1 + np.exp(x),
        'leaky_relu': lambda x: np.max([x, 0.1*x], 0)
        # 'softmax': lambda x:
    }[activation_fn]
    self.d_activation_fn = {
        'sigmoid': lambda x: self.activation_fn(x) * (1 - self.activation_fn(x)),
        'leaky_relu': lambda x: 1 if x > 0.0 else 0.1
    }[activation_fn]
    self.output_unit = output_unit
    self.prev_layer = None
    self.next_layer = None
    self.velocity = None
    self.output = None
    self.weight = None
    self.bias = None
  def __call__(self, prev_layer):
    self.prev_layer = prev_layer
    self.prev_layer.next_layer = self
    self.weight = np.zeros((self.prev_layer.output_unit, self.output_unit))
    if self.weight_random_init:
      self.weight += np.random.uniform(0.0, 0.1, (self.prev_layer.output_unit, self.output_unit))
    self.bias = np.zeros((1, self.output_unit))
    if self.bias_random_init:
      self.bias += np.random.uniform(0.0, 0.1, (1, self.output_unit))
    self.velocity = np.zeros_like(self.weight)
  def _feed_fwd(self):
    # out^T = x^T * W^T + b^T
    self.output = np.matmul(self.prev_layer.output, self.weight) + self.bias
    self.output = self.activation_fn(self.output)
    if self.next_layer is not None:
      self.next_layer._feed_fwd()
  def _back_prop(self, dError_dOut, learning_rate, momentum=0.0):
    dOut_dNet = self.d_activation_fn(self.output)
    dNet_dW = self.prev_layer.output
    dError_dNet = dError_dOut * dOut_dNet
    dError_dW = np.matmul(np.insert(dNet_dW.T, 0, 1., axis=1), dError_dNet)
    dError_dOut_prev = np.matmul(dError_dNet, self.weight.T)
    v = momentum * self.velocity + learning_rate * dError_dW
    x = -momentum * self.velocity + (1 + momentum) * v
    bias_x, weight_x = np.split(x, [1], axis=1)
    self.weight += weight_x
    self.bias += bias_x.T
    self.velocity = v
    self._back_prop(dError_dOut_prev, learning_rate, momentum)
class Model:
  def __init__(self):
    self.data_manager = DataManager('./twitter-sentiment.csv', do_cleaning=True)
    self.input_layer = None
    self.output_layer = None
    self.loss_fn = None
    self.gradient_fn = None
    self._build_model()
  def _build_model(self):
    self.loss_fn, self.gradient_fn = self.cross_entropy()
    self.input_layer = Input(self.data_manager.vec_len)
    self.output_layer = Dense(1, 'sigmoid', self.data_manager.vec_len)(self.input_layer)
  def fit(self, data, label, batch_size=1, epoch=1, log_file='run.log', split_ratio=0.2):
    X_train, X_valid, Y_train, Y_valid = train_test_split(data, label,
                                                          test_size=split_ratio, random_state=0)
    # TODO: shuffle, main learning loop, write log to file
  def predict(self, X):
    # TODO: implement later
    pass
  @staticmethod
  def cross_entropy():
    def loss(target, output):
      batch_loss = np.sum(target * np.log(output) + (1. - target) * np.log(1. - output),
                          axis=1, keepdims=True)
      return np.mean(batch_loss, axis=0)
    def gradient(target, output):
      dE_dOut = -(target / output + (1. - target) / (1. - output))
      return dE_dOut
    return loss, gradient

# def compute_loss(A, Y, m):
#   """
#   TODO:
#     Compute the loss function based on the formula you derived.
#     loss is a scalar
#     Hint:
#       1) The formula should be (-1.0 / m) * np.sum(...)
#   """
#   loss = 0
#   return loss

# def back_prop(X, A, Y, m):
#   """
#   TODO:
#     Compute the gradient based on the formula you derived.
#     dw and db are two scalars.
#     Hint:
#       1) The formula of dw should be (1.0 / m) * np.dot(...)
#       2) The formula of db should be (1.0 / m) * np.sum(...)
#   """
#   dw = 0
#   db = 0
#   return {"dw": dw, "db": db}

# def optimize(w, b, X, Y, X_dev, Y_dev, num_iterations, learning_rate, output_name):
#   m = X.shape[1] # m is the number of the samples
#   max_acc = 0
#   max_w, max_b = w, b
#   start_time = time.time()
#   log = open(output_name+'.log', 'w')
#   log.write('iteration, train acc, dev acc\n')

#   for i in range(num_iterations):
#     f_x = forward_prop(X, w, b)
#     cost = compute_loss(f_x, Y, m)
#     grads = back_prop(X, f_x, Y, m)

#     w = w - learning_rate * grads["dw"]
#     b = b - learning_rate * grads["db"]

#     Y_prediction_train = predict(w, b, X)
#     Y_prediction_dev = predict(w, b, X_dev)
#     train_acc = compare(Y_prediction_train, Y)
#     dev_acc = compare(Y_prediction_dev, Y_dev)
#     log.write('{},{},{}\n'.format(str(i+1), str(train_acc), str(dev_acc)))

#     if dev_acc > max_acc: # keep the best parameters
#       mac_acc = dev_acc
#       max_w, max_b = w, b

#     print('iteration:', i+1, ", time {0:.2f}", time.time()-start_time)
#     print("\tTraining accuracy: {0:.4f} %, cost: {0:.4f}".format(train_acc, cost))
#     print("\tDev accuracy: {0:.4f} %".format(dev_acc))

#   params = {"w": max_w,
#             "b": max_b}
#   return params

# def predict(w, b, X):
#   """
#   TODO: 
#     Predict the sentiment class based on the f(x) value. 
#     if f(x) > 0.5, then pred value is 1, otherwise is 0.
#     Y_prediction is a 2-D array with the size (1*nb_sentence)
#   """  
#   m = X.shape[1]
#   Y_prediction = np.zeros((1, m))
#   return Y_prediction

# def compare(pred, gold):
#   """
#   TODO: 
#     Compute the accuracy based on two array, pred and gold, and return a scalar between [0, 100]
#   """  
#   acc = 0
#   return acc

# def write_testset_prediction(parameters, test_data, file_name="myPrediction.csv"):
#   Y_prediction_test = predict(parameters['w'], parameters['b'], test_data)
#   f_pred = open(file_name, 'w')
#   f_pred.write('ID\tSentiment')
#   ID = 1
#   for pred in Y_prediction_test[0]:
#     sentiment_pred = 'pos' if pred==1 else 'neg'
#     f_pred.write(str(ID)+','+sentiment_pred+'\n')
#     ID += 1

# def model(X_train, Y_train, X_dev, Y_dev, output_name, num_iterations=100, learning_rate=0.005):
#   w, b = initialize_w_and_b(X_train.shape[0])

#   parameters = optimize(w, b, X_train, Y_train, X_dev, Y_dev, num_iterations, learning_rate, output_name)

#   Y_prediction_dev = predict(parameters["w"], parameters["b"], X_dev)
#   print("Best dev accuracy: {} %".format(compare(Y_prediction_dev, Y_dev)))

#   np.save(output_name+'.npy', parameters)

#   return parameters

# if __name__ == "__main__":
#   import argparse
#   parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
#   parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
#   parser.add_argument('-mv','--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
#   parser.add_argument('-lr','--learning_rate', required=False, default=0.1)
#   parser.add_argument('-i','--num_iter', required=False, default=200)
#   parser.add_argument('-fn','--file_name', help='file name', required=False, default='myTest')
#   args = vars(parser.parse_args())
#   print(args)

#   print('[Read the data from twitter-sentiment.csv...]')
#   revs, word2idx = data_preprocess('./twitter-sentiment.csv', args['clean'], int(args['max_vocab']))
  
#   print('[Extract features from the read data...]')
#   data, label = feature_extraction_bow(revs, word2idx)
#   data = normalization(data)
  
#   # shuffle data
#   shuffle_idx = np.arange(len(data))
#   np.random.shuffle(shuffle_idx)
#   data = data[shuffle_idx]
#   label = label[shuffle_idx]

#   print('[Start training...]')
#   X_train, X_dev, Y_train, Y_dev = train_test_split(data, label, test_size=0.2, random_state=0)
#   parameters = model(X_train.T, Y_train.T, X_dev.T, Y_dev.T, args['file_name'], 
#             num_iterations=int(args['num_iter']), learning_rate=float(args['learning_rate']))
  
#   print('\n[Start evaluating on the official test set and dump as {}...]'.format(args['file_name']+'.csv'))
#   revs, _ = data_preprocess("./twitter-sentiment-testset.csv", args['clean'], int(args['max_vocab']))
#   test_data, _ = feature_extraction_bow(revs, word2idx)
#   write_testset_prediction(parameters, test_data.T, args['file_name']+'.csv' )
