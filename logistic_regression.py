import re
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocess import DataManager

EPSILON = 1e-8

class Input:
  """
  Input layer
  """
  def __init__(self, input_dim):
    """
    Args:
      input_dim:          Input vector demension
    """
    self.output_unit = input_dim
    self.output = None
    self.next_layer = None
  def __call__(self, _input):
    self.output = _input
    self.next_layer._feed_fwd()
    return self
  def _back_prop(self, dError_dOut, learning_rate, momentum=0.):
    # Do nothing
    pass
  def _save(self, h5_file):
    if 'input' in h5_file.keys():
      h5_file['input'] = self.output_unit
    else:
      h5_file.create_dataset('input', data=self.output_unit)
    self.next_layer._save(h5_file, 0)
class Dense:
  """
  Dense layer
  """
  def __init__(self, output_unit, activation_fn_name,
               bias_random_init=False, weight_random_init=False):
    """
    Args:
      output_unit:        Number of neuron
      activation_fn_name: Activation function name
      bias_random_init:   Indicate whether bias should be initialized
                            uniformly distributed [0., .1)
      weight_random_init: Indicate whether weight should be initialized
                            uniformly distributed [0., .1)
    """
    self.bias_random_init = bias_random_init
    self.weight_random_init = weight_random_init
    assert isinstance(activation_fn_name, str)
    self.activation_fn_name = activation_fn_name
    self.activation_fn = {
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'leaky_relu': lambda x: np.max([x, 0.1*x], 0)
        # 'softmax': lambda x:
    }[activation_fn_name]
    self.d_activation_fn = {
        'sigmoid': lambda y: y * (1 - y),
        'leaky_relu': lambda y: 1 if y > 0. else .1
    }[activation_fn_name]
    self.output_unit = output_unit
    self.prev_layer = None
    self.next_layer = None
    self.velocity = None
    self.output = None
    self.weight = None
    self.bias = None
  def __call__(self, prev_layer):
    # Set up connection between adjacent layers
    self.prev_layer = prev_layer
    self.prev_layer.next_layer = self
    # Initialize weight
    self.weight = np.zeros((self.prev_layer.output_unit, self.output_unit))
    if self.weight_random_init:
      self.weight += np.random.uniform(0., .1, (self.prev_layer.output_unit, self.output_unit))
    # Initialize bias
    self.bias = np.zeros((1, self.output_unit))
    if self.bias_random_init:
      self.bias += np.random.uniform(0., .1, (1, self.output_unit))
    # Initialize velocity for nesterov momentum
    self.velocity = np.zeros((self.prev_layer.output_unit+1, self.output_unit))
    return self
  def _feed_fwd(self):
    # out = Wx + b
    # With batch size version,
    #   out^T = (x^T)(W^T) + b^T
    self.output = np.matmul(self.prev_layer.output, self.weight) + self.bias
    # Pass through activation function
    self.output = self.activation_fn(self.output)
    # Forward to the next layer if it exists
    if self.next_layer is not None:
      self.next_layer._feed_fwd()
  def _back_prop(self, dError_dOut, learning_rate, momentum=0.):
    # dError/dW = dError/dOut * dOut/dNet * dNet/dW
    #   where Out = activation_function(Net = Out_prev * W + b)
    dOut_dNet = self.d_activation_fn(self.output)
    dNet_dW = self.prev_layer.output
    dError_dNet = dError_dOut * dOut_dNet
    dError_dW = np.matmul(np.insert(dNet_dW.T, 0, 1., axis=0), dError_dNet)
    # dError/dOut_prev = dError/dNet * dNet/dOut_prev
    dError_dOut_prev = np.matmul(dError_dNet, self.weight.T)
    # Compute velocity
    batch_size = np.shape(dNet_dW)[0]
    v = momentum * self.velocity + learning_rate * dError_dW / batch_size
    x = -momentum * self.velocity + (1. + momentum) * v
    self.velocity = v
    # Update weight and bias
    bias_x, weight_x = np.split(x, [1], axis=0)
    self.weight -= weight_x
    self.bias -= bias_x.T
    # Propagation error to previous layer
    self.prev_layer._back_prop(dError_dOut_prev, learning_rate, momentum)
  def _save(self, h5_file, index):
    if '%d_dense_w'%(index) in h5_file.keys():
      h5_file['%d_dense_w'%(index)] = self.weight
      h5_file['%d_dense_b'%(index)] = self.bias
      h5_file['%d_dense_v'%(index)] = self.velocity
      h5_file['%d_dense_a'%(index)] = self.activation_fn_name
      h5_file['%d_dense_o'%(index)] = self.output_unit
    else:
      assert any([not x.startswith('%d_'%(index)) or x.startswith('%d_dense'%(index)) for x in h5_file.keys()])
      h5_file.create_dataset('%d_dense_w'%(index), data=self.weight)
      h5_file.create_dataset('%d_dense_b'%(index), data=self.bias)
      h5_file.create_dataset('%d_dense_v'%(index), data=self.velocity)
      h5_file.create_dataset('%d_dense_a'%(index), data=self.activation_fn_name)
      h5_file.create_dataset('%d_dense_o'%(index), data=self.output_unit)
    if self.next_layer is not None:
      self.next_layer._save(h5_file, index+1)
  @staticmethod
  def _load(h5_file, index):
    # TODO: test this function
    related_key_list = [x for x in h5_file.keys() if x.startsWith(index+'_')]
    var_list = {re.search(r'dense_(.+)$', key)[1]: h5_file[key] for key in related_key_list}
    self_obj = Dense(var_list['o'], var_list['a'])
    self_obj.weight = var_list['w']
    self_obj.bias = var_list['b']
    self_obj.velocity = var_list['v']
    return self_obj
class Model:
  """
  Manage model
  """
  def __init__(self, learning_rate, momentum=0.):
    """
    Args:
      learning_rate:      For update weight and bias
      momentum:           For update velocity [Recommend: .9]
    """
    self.learning_rate = learning_rate
    self.momentum = momentum
    # TODO: path should be passed from outside?
    self.data_manager = DataManager('./twitter-sentiment.csv', do_cleaning=True)
    self.input_layer = None
    self.output_layer = None
    self.loss_fn = None
    self.gradient_fn = None
    self._build_model()
  def _build_model(self):
    # Get functions for evaluating loss, gradient and accuarcy
    self.loss_fn, self.gradient_fn, self.accuracy_fn = self.cross_entropy()
    # Input layer
    self.input_layer = Input(self.data_manager.input_vec_len)
    # Output layer:
    #   activation func:                      sigmoid
    #   Bias, weight random initialization:   True
    self.output_layer = Dense(1, 'sigmoid', True, True)(self.input_layer)
  def fit(self, batch_size=1, epoch=1, log_file='run.log', split_ratio=.2, model_file=None):
    """
    Train the network for a fixed number of epochs

    Args:
      batch_size:           Determine how many samples are used in a single iteration
      epoch:                Determine how many times the whole dataset is used for training
      log_file:             Log file path [if the file exists, it'll overwrite.
                                           if not, it creates one]
      split_ratio:          Ratio of validation set to training set
      model_file:           File for saving model having the lowest validation loss
    """
    # Get data and label
    data, label = self.data_manager.extract_feature(True)
    # Split into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(data, label,
                                                          test_size=split_ratio, random_state=0)
    # Delete original variables for saving memory  
    del data, label

    # Check whether the log file has correct file extension
    if not log_file.endswith('.log'):
      log_file += '.log'

    # Main loop
    best_valid_loss = 999.
    best_valid_acc = None
    print('Start training...')
    with open(log_file, 'w+') as log:
      # Write header to log file
      log.write('iteration, train acc, dev acc\n')
      # For each epoch
      for epoch_index in range(epoch):
        sum_train_loss = 0.
        train_correct_count = 0.
        print('\r[ %d / %d ] epoch:'%(epoch_index + 1, epoch))
        print('\r[%s] (%3.f%%) loss: %.4f'%('-'*20, 0., 0.), end='\033[K')
        rand_ints = np.random.permutation(len(x_train))
        total_train_size = len(rand_ints) // batch_size * batch_size
        for iter_index in range(len(rand_ints) // batch_size):
          # Shuffle data and label
          data_batch = x_train[rand_ints[iter_index*batch_size:(iter_index+1)*batch_size]]
          label_batch = y_train[rand_ints[iter_index*batch_size:(iter_index+1)*batch_size]]
          # Feed forward
          self.input_layer(data_batch)
          # Calculate loss
          sum_train_loss += self.loss_fn(label_batch, self.output_layer.output)
          train_correct_count += self.accuracy_fn(label_batch,
                                                  self.output_layer.output) * len(label_batch)
          # Back propagation
          gradient = self.gradient_fn(label_batch, self.output_layer.output)
          self.output_layer._back_prop(gradient, self.learning_rate, self.momentum)
          # Print loss and accuracy
          progress = min((iter_index + 1) / (len(rand_ints) // batch_size) * 20., 20.)
          print('\r[%s] (%3.f%%) loss: %.4f acc: %.3f'%
                ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                 sum_train_loss / (iter_index + 1),
                 train_correct_count / ((iter_index+1) * batch_size)), end='\033[K')
        # Calculate validation loss and accuracy
        sum_valid_loss = 0.
        valid_correct_count = 0.
        rand_ints = np.random.permutation(len(x_valid))
        total_valid_size = len(rand_ints) // batch_size * batch_size
        for iter_index in range(len(rand_ints) // batch_size):
          data_batch = x_valid[rand_ints[iter_index*batch_size:(iter_index+1)*batch_size]]
          label_batch = y_valid[rand_ints[iter_index*batch_size:(iter_index+1)*batch_size]]
          self.input_layer(data_batch)
          sum_valid_loss += self.loss_fn(label_batch, self.output_layer.output)
          valid_correct_count += self.accuracy_fn(label_batch,
                                                  self.output_layer.output) * len(label_batch)
        mean_valid_loss = sum_valid_loss / total_valid_size
        mean_valid_acc = valid_correct_count / total_valid_size
        print(' val_loss: %.4f val_acc: %.3f'%(mean_valid_loss, mean_valid_acc))
        log.write('%d,%.5f,%.5f\n'%(epoch_index, train_correct_count / total_train_size,
                                    mean_valid_acc))
        if mean_valid_loss < best_valid_loss and model_file is not None:
          best_valid_loss = mean_valid_loss
          self.save(model_file, best_valid_loss)
          print(' -- saved', end='')
          best_train_acc = mean_valid_acc
        print('\n')
    print('\nBest validation accuracy: %d\n'%(best_train_acc))
    print('Finish training!\n')
  def save(self, file_name, valid_loss):
    if not file_name.endswith('.h5'):
      file_name += '.h5'
    import h5py
    with h5py.File(file_name, 'w') as h5f:
      if 'best_valid_loss' not in h5f.keys() or h5f['best_valid_loss'] > valid_loss:
        self.input_layer._save(h5f)
        h5f['best_valid_loss'] = valid_loss
  def load(self, file_name):
    if not file_name.endswith('.h5'):
      file_name += '.h5'
    import h5py
    with h5py.File(file_name, 'r') as h5f:
      var_name_list = sorted(h5f.keys())
      last_index = -1
      last_layer = None
      last_layer = self.input_layer = Input(h5f['input'])
      for var_name in var_name_list:
        if not var_name.startswith(last_index+'_'):
          search_result = re.search(r'([0-9]+)_([^_]+)_.+', var_name)
          last_index = int(search_result[1])
          class_name = search_result[2]
          new_layer = globals[class_name]._load(h5f, last_index)
          new_layer(last_layer)
          last_layer = new_layer
      self.output_layer = last_layer
  @staticmethod
  def cross_entropy():
    def loss(target, output):
      batch_loss = -np.sum(target * np.log(output+EPSILON) \
                           + (1.-target) * np.log(1.-output+EPSILON),
                           axis=1, keepdims=True)
      return np.mean(batch_loss, axis=0)
    def gradient(target, output):
      dE_dOut = -target / (output+EPSILON) + (1.-target) / (1.-output+EPSILON)
      return dE_dOut
    def accuracy(target, output):
      _output = output.copy()
      shape = np.shape(output)
      _output[_output < .5] = 0.
      _output[_output >= .5] = 1.
      correct = np.sum(np.equal(target, _output).astype(np.float64), axis=1)
      correct[correct != shape[1]] = 0.
      correct[correct == shape[1]] = 1.
      return np.mean(correct)
    return loss, gradient, accuracy

if __name__ == '__main__':
  model = Model(.001, .9)
  model.fit(batch_size=50, epoch=10, model_file='model')

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
