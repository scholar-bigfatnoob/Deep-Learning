# Lecture Series I (Introduction)

### Softmax
* Used to normalize predicted outputs from 
different hidden layers or output layer.
* Large values get closer to 1 and small values get closer to 0.
* Mathematically, `S(x_i) = exp(x_i)/∑exp(x)`
* Programatically,
```
def softmax(x):
  exps = np.exp(x)
  return exps / np.sum(exps, 0)
```
* For values at a larger scale, softmax generates polar vectors(closer to 0 and 1).
* For values at a smaller scale, softmax generates uniform vectors.
 

### One Hot Encoding(OHE)
* Used in multi-class classification. 
* Converts multi-classes into a vector where the actual class is denoted by 1 and all other classes are denoted by 0.

### Cross Entropy
* Used to predict similarity between predicted and actual class.
* Mathematically `D(S,L) = -∑L_i * log(S_i)` where `S` is the predicted vector and `L` is the actual OHE vector.
* This is used to compute loss which can be further used for optimization.
* Programatically,
```
def cross_entropy(predicted, labels):
  return -1 * np.sum(labels * np.log(predicted), axis=0)
```
* The value of the resultant loss, is used to update the weights along with a learning rate.

### Inputs
* Initial inputs should be normalized between -1 and 1.  
* Roughly with 0 mean and equal variances.

### Weights
* Choice of initial weights crucial in learning the NN, DNN
* Initial weights should be randomly chosen from a gaussian distribution with 0 mean and a small variance.
* If a large variance is used, the model might peak too soon resulting in over-fitting and poor accuracy on the test set.
 
### Validation
* Use a validation set for hyper parameter optimization.
* The training set should be split to create a validation set.
* As the model is optimized, the model starts to overfit with the validation set as well.
* To avoid this the model should be tested on a set of datapoints which are not part of the training or validation set.
* Alternatively, cross validation can be used.
