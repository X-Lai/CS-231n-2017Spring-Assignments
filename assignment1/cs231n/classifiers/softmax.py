import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  ys = np.zeros((num_train, num_classes))
  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    max_score = np.max(scores)
    scores_exp = np.exp(scores)
    minus_max_score_exp = np.exp(-max_score)
    sum_scores_exp = np.sum(scores_exp)
    for j in xrange(num_classes):
        ys[i][j] = (scores_exp[j]*minus_max_score_exp) / (sum_scores_exp * minus_max_score_exp)
    for j in xrange(num_classes):
        T = X[i]/ys[i][y[i]]
        P = ys[i][j]
        if j == y[i]:
            dW[:,j] += (P-P*P)*T
        else:
            dW[:,j] += (-P*ys[i][y[i]])*T
  loss = np.sum(-np.log(ys[range(num_train), y])) / num_train + reg * np.sum(W*W)
  dW = -dW/num_train
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  scores_exp = np.exp(scores)
  scores_max = np.max(scores, axis=1).reshape(num_train,1)
  scores_exp_sum = np.sum(scores_exp, axis=1).reshape(num_train,1)
  minus_scores_max_exp = np.exp(-scores_max).reshape(num_train,1)
  ys = scores_exp * minus_scores_max_exp / (scores_exp_sum * minus_scores_max_exp)
  loss = np.sum(-np.log(ys[range(num_train), y])) / num_train + reg*np.sum(W*W)
  T = ys[range(num_train), y]
  ds = -ys*(T.reshape(num_train, 1))
  ds[range(num_train), y] += T
  dW = np.dot(np.transpose(X/(T.reshape(num_train,1))), ds)
  dW = -dW/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

