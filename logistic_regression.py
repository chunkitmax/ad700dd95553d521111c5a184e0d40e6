'''
logistic_regression.py

Model training
'''
import os
import os.path as Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import DataManager

EPSILON = 1e-8

def add_epsilon(v):
  v[v == 0.] = EPSILON
  return v

class Activation:
  @staticmethod
  def sigmoid():
    def _sigmoid(x):
      return 1 / (1 + np.exp(-x))
    def _d_sigmoid(y):
      return y * (1 - y)
    return _sigmoid, _d_sigmoid
  @staticmethod
  def leaky_relu():
    def _leaky_relu(x):
      return np.max([x, 0.1*x], 0)
    def _d_leaky_relu(y):
      _y = np.zeros_like(y, dtype=np.float64)
      _y[y >= 0.] = 1.
      return _y
    return _leaky_relu, _d_leaky_relu

class Loss:
  @staticmethod
  def cross_entropy():
    def loss(target, output):
      batch_loss = -np.sum(target * np.log(add_epsilon(output)) \
                           + (1.-target) * np.log(add_epsilon(1.-output)),
                           axis=1, keepdims=True)
      return np.mean(batch_loss, axis=0)
    def gradient(target, output):
      dE_dOut = -target / (add_epsilon(output)) + (1.-target) / add_epsilon(1.-output)
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
      activation_fn_name:      Activation function name
      bias_random_init:   Indicate whether bias should be initialized
                            uniformly distributed [0., .1)
      weight_random_init: Indicate whether weight should be initialized
                            uniformly distributed [0., .1)
    """
    self.bias_random_init = bias_random_init
    self.weight_random_init = weight_random_init
    assert isinstance(activation_fn_name, str)
    self.activation_fn_name = activation_fn_name
    self.activation_fn, self.d_activation_fn = getattr(Activation, self.activation_fn_name)()
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
    self.bias -= bias_x
    # Propagation error to previous layer
    self.prev_layer._back_prop(dError_dOut_prev, learning_rate, momentum)
  def _save(self, h5_file, index):
    if '%d_Dense_w'%(index) in h5_file.keys():
      h5_file['%d_Dense_w'%(index)] = self.weight
      h5_file['%d_Dense_b'%(index)] = self.bias
      h5_file['%d_Dense_v'%(index)] = self.velocity
      h5_file['%d_Dense_a'%(index)] = self.activation_fn_name
      h5_file['%d_Dense_o'%(index)] = self.output_unit
    else:
      assert any([not x.startswith('%d_'%(index)) or \
                 x.startswith('%d_dense'%(index)) for x in h5_file.keys()])
      h5_file.create_dataset('%d_Dense_w'%(index), data=self.weight)
      h5_file.create_dataset('%d_Dense_b'%(index), data=self.bias)
      h5_file.create_dataset('%d_Dense_v'%(index), data=self.velocity)
      h5_file.create_dataset('%d_Dense_a'%(index), data=self.activation_fn_name)
      h5_file.create_dataset('%d_Dense_o'%(index), data=self.output_unit)
    if self.next_layer is not None:
      self.next_layer._save(h5_file, index+1)
  @staticmethod
  def _load(h5_file, index, prev_layer):
    related_key_list = [x for x in h5_file.keys() if x.startswith(str(index)+'_')]
    var_list = {re.search(r'Dense_(.+)$', key)[1]: h5_file[key][()] for key in related_key_list}
    self_obj = Dense(var_list['o'], var_list['a'])
    self_obj.weight = var_list['w']
    self_obj.bias = var_list['b']
    self_obj.velocity = var_list['v']
    self_obj.prev_layer = prev_layer
    prev_layer.next_layer = self_obj
    return self_obj
class Model:
  """
  Manage model
  """
  def __init__(self, input_dim, learning_rate, momentum=0.,
               model_file=None, load_model=False):
    """
    Args:
      input_dim:          Length of input vector
      learning_rate:      For update weight and bias
      momentum:           For update velocity [Recommend: .9]
    """
    self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.input_layer = None
    self.output_layer = None
    self.loss_fn = None
    self.gradient_fn = None
    self.model_file = model_file
    if self.model_file is not None:
      if not self.model_file.endswith('.h5'):
        self.model_file += '.h5'
      if load_model and Path.exists(self.model_file):
        print('\n\nModel loaded\n')
        self._load_model()
        return
    self._build_model()
    print('\n\nModel built\n')
  def _build_model(self):
    # Get functions for evaluating loss, gradient and accuarcy
    self.loss_fn, self.gradient_fn, self.accuracy_fn = Loss.cross_entropy()
    # Input layer
    self.input_layer = Input(self.input_dim)
    # Output layer:
    #   activation func:                      sigmoid
    #   Bias, weight random initialization:   True
    self.output_layer = Dense(1, 'sigmoid', True, True)(self.input_layer)
  def _load_model(self):
    self.loss_fn, self.gradient_fn, self.accuracy_fn = Loss.cross_entropy()
    import h5py
    with h5py.File(self.model_file, 'r') as h5f:
      var_name_list = sorted(h5f.keys())
      last_index = -1
      last_layer = None
      last_layer = self.input_layer = Input(h5f['input'][()])
      for var_name in var_name_list:
        if var_name.startswith(str(last_index+1)+'_'):
          search_result = re.search(r'[0-9]+_([^_]+)_.+', var_name)
          last_index += 1
          class_name = search_result[1]
          new_layer = globals()[class_name]._load(h5f, last_index, last_layer)
          last_layer = new_layer
      self.output_layer = last_layer
  def fit(self, X, Y, batch_size=1, epoch=1, log_file='run.log', split_ratio=.2, plot=False):
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
    # Split into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
                                                          test_size=split_ratio, random_state=0)
    # Delete original variables for saving memory
    # del X, Y

    # Check whether the log file has correct file extension
    if not log_file.endswith('.log'):
      log_file += '.log'

    if plot:
      import matplotlib.pyplot as plt
      from scipy.interpolate import spline
      fig, (ax1, ax2) = plt.subplots(2, 1)
      ax1.set_xlabel('epoch')
      ax1.set_ylabel('value')
      ax2.set_xlabel('epoch')
      ax2.set_ylabel('value')

    # Main loop
    best_valid_loss = 999.
    best_valid_acc = 0.
    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    try:
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
                   sum_train_loss / ((iter_index + 1) * batch_size),
                   train_correct_count / ((iter_index+1) * batch_size)), end='\033[K')
          train_loss_history.append(sum_train_loss / total_train_size)
          mean_train_acc = train_correct_count / total_train_size
          train_acc_history.append(mean_train_acc)
          # Calculate validation loss and accuracy
          sum_valid_loss = 0.
          valid_correct_count = 0.
          total_valid_size = len(x_valid) // batch_size * batch_size
          for iter_index in range(len(x_valid) // batch_size):
            data_batch = x_valid[iter_index*batch_size:(iter_index+1)*batch_size]
            label_batch = y_valid[iter_index*batch_size:(iter_index+1)*batch_size]
            self.input_layer(data_batch)
            sum_valid_loss += self.loss_fn(label_batch, self.output_layer.output)
            valid_correct_count += self.accuracy_fn(label_batch,
                                                    self.output_layer.output) * len(label_batch)
          mean_valid_loss = sum_valid_loss / total_valid_size
          mean_valid_acc = valid_correct_count / total_valid_size
          valid_loss_history.append(mean_valid_loss)
          valid_acc_history.append(mean_valid_acc)
          print(' val_loss: %.4f val_acc: %.3f'%(mean_valid_loss, mean_valid_acc), end='')
          log.write('%d,%.5f,%.5f\n'%(epoch_index, mean_train_acc, mean_valid_acc))
          # Save best model
          # if mean_valid_loss < best_valid_loss:
          #   best_valid_loss = mean_valid_loss
          #   best_valid_acc = mean_valid_acc
          #   if self.model_file is not None:
          #     self.save(self.model_file, best_valid_loss)
          if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            best_valid_loss = mean_valid_loss
            if self.model_file is not None:
              self.save(self.model_file, best_valid_acc)
          print('\n')
          if plot and len(train_loss_history) > 3:
            x_axis = np.arange(len(train_loss_history))
            smooth_x = np.linspace(0, len(train_loss_history)-1)
            smooth_train_loss_history = spline(x_axis, np.squeeze(train_loss_history), smooth_x)
            smooth_train_acc_history = spline(x_axis, np.squeeze(train_acc_history), smooth_x)
            smooth_valid_loss_history = spline(x_axis, np.squeeze(valid_loss_history), smooth_x)
            smooth_valid_acc_history = spline(x_axis, np.squeeze(valid_acc_history), smooth_x)
            ax1.clear()
            ax2.clear()
            ax1.plot(smooth_x, smooth_train_loss_history, 'm-',
                     smooth_x, smooth_valid_loss_history, 'r-')
            ax2.plot(smooth_x, smooth_train_acc_history, 'c-',
                     smooth_x, smooth_valid_acc_history, 'b-')
            ax1.legend(['train_loss', 'valid_loss'])
            ax2.legend(['train_acc', 'valid_acc'])
            fig.canvas.draw()
            fig.show()
    except KeyboardInterrupt:
      print('Training interrupted...\n')
      pass
    print('\nBest validation accuracy: %d\n'%(best_valid_acc))
    if plot:
      fig.savefig('result.png')
    print('Finish training!\n')
  def save(self, file_name, valid_loss):
    if not file_name.endswith('.h5'):
      file_name += '.h5'
    import h5py
    with h5py.File(file_name, 'w') as h5f:
      # if 'best_valid_loss' not in h5f.keys() or h5f['best_valid_loss'] > valid_loss:
      #   self.input_layer._save(h5f)
      #   h5f['best_valid_loss'] = valid_loss
      #   print(' -- saved', end='')
      if 'best_valid_acc' not in h5f.keys() or h5f['best_valid_acc'] > valid_loss:
        self.input_layer._save(h5f)
        h5f['best_valid_acc'] = valid_loss
        print(' -- saved', end='')
  def predict(self, X):
    num_batch = len(X) // 50
    if len(X) - num_batch * 50 > 0:
      num_batch += 1
    results = []
    for i in range(num_batch):
      data_batch = X[i*50:(i+1)*50]
      # Feed forward
      self.input_layer(data_batch)
      result = np.zeros_like(self.output_layer.output, dtype=np.int32)
      result[self.output_layer.output >= .5] = 1
      results.append(result)
    return np.reshape(results, (-1))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
  parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False',
                      required=False, default=False, action='store_true')
  parser.add_argument('-n', '--normalize', help='Normalize input vectors, default is False',
                      required=False, default=True, action='store_true')
  parser.add_argument('-p', '--plot', help='Plot results, default is False',
                      required=False, default=False, action='store_true')
  parser.add_argument('-mv', '--max_vocab', help='Max vocab size predifined, no limit if set None or <0',
                      required=False, default=None)
  parser.add_argument('-ng', '--ngram', help='ngram predifined, default is 1(=unigram)',
                      required=False, default=1, type=int)
  parser.add_argument('-lr', '--learning_rate', required=False, default=0.001, type=float)
  parser.add_argument('-m', '--momentum', required=False, default=0.9, type=float)
  parser.add_argument('-i', '--num_iter', required=False, default=10, type=int)
  parser.add_argument('-b', '--batch_size', help='Batch size, default is 50',
                      required=False, default=50, type=int)
  parser.add_argument('-dfn', '--data_file_name',
                      help='Data file name, default is "twitter-sentiment"',
                      required=False, default='twitter-sentiment')
  parser.add_argument('-tfn', '--test_file_name',
                      help='Test file name, default is "twitter-sentiment-testset"',
                      required=False, default='twitter-sentiment-testset')
  parser.add_argument('-fn', '--file_name', help='Output file name, default is "myTest"',
                      required=False, default='myTest')
  parser.add_argument('-mfn', '--model_file_name',
                      help='Model file name, default is None',
                      required=False, default=None)
  parser.add_argument('-l', '--load', help='Load model file, default is False',
                      required=False, default=False, action='store_true')
  args = vars(parser.parse_args())

  # Train network
  data_manager = DataManager(args['data_file_name'], do_cleaning=args['clean'],
                             max_vocab_size=args['max_vocab'], ngram=args['ngram'])
  if args['model_file_name'] is not None:
    model = Model(data_manager.input_vec_len, args['learning_rate'], args['momentum'],
                  model_file=args['model_file_name'], load_model=args['load'])
  else:
    model = Model(data_manager.input_vec_len, args['learning_rate'], args['momentum'],
                  model_file='temp_model', load_model=False)
  X, Y = data_manager.extract_feature(do_normailzation=args['normalize'])
  model.fit(X, Y, batch_size=args['batch_size'], epoch=args['num_iter'], plot=args['plot'])
  del X, Y, model

  # Predict data
  X, _ = data_manager.extract_feature(do_normailzation=args['normalize'],
                                      target_file_name=args['test_file_name'])
  if args['model_file_name'] is not None:
    model = Model(data_manager.input_vec_len, args['learning_rate'], args['momentum'],
                  model_file=args['model_file_name'], load_model=True)
  else:
    model = Model(data_manager.input_vec_len, args['learning_rate'], args['momentum'],
                  model_file='temp_model', load_model=True)
  prediction = model.predict(X)
  if not args['file_name'].endswith('.csv'):
    args['file_name'] += '.csv'
  with open(args['file_name'], 'w+') as prediction_file:
    prediction_file.write('ID\tsentiment\n')
    bool_str = ['neg', 'pos']
    lines = ['%d\t%s\n'%(index+1, bool_str[result]) for index, result in enumerate(prediction)]
    prediction_file.writelines(lines)
    print('\nAll finished!\n')

  # Clean up temp file
  if args['model_file_name'] is None and Path.exists('temp_model.h5'):
    os.remove('temp_model.h5')
  del X, data_manager
