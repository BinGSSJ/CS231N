import numpy as np
from random import shuffle

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

  for i in range(num_train):
    scores = X[i].dot(W)#第一个样本的结果 
    shift_scores = scores - np.max(scores)
    Syi = shift_scores[y[i]]
    loss += -np.log((np.exp(Syi) / np.sum(np.exp(shift_scores))))
    for j in range(num_classes):
      softmax_out = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))
      if j == y[i]:
        dW[:,j] += X[i].T * (softmax_out - 1)
      else:
        dW[:,j] += X[i].T * (softmax_out)

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis=1).reshape(-1,1)
  correct_class = np.exp(shift_scores[np.arange(num_train), y])
  tmp = np.sum(np.exp(shift_scores), axis=1)#每个样本的e的Sj次方的和
  loss = np.sum(-np.log(correct_class / tmp))
  dS = np.zeros_like(scores)
  dS = np.exp(shift_scores) / tmp.reshape(num_train, 1)
  dS[np.arange(num_train), y] -= 1#j=yi时原有梯度上减1

  dW = X.T.dot(dS)
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

