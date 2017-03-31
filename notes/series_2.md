# Lecture Series II (Deep Neural Networks)

### ReLU
* Stands for Rectified linear unit.
* Mathematically, If x is less than 0, y is 0 else y = x.
* Programatically
```
def relu(x):
  return x if x > 0 else 0
```
* Derivative of Relu is 0 if x < 0 else it is 1.

### Getting rid of linearity.
* Simple Logistic regression models are linear in nature which does not
account for the multiplicative dependence between the independent and 
dependent variables.
* Thus a combination of non-linear transformations can be applied between 
 the input and output layer to account for such dependencies.
* This is done using ReLU(s). Outputs from the input layer is sent to a ReLU 
which is then sent through another set of weights and finally sent out through
a softmax function. This becomes a hidden layer.


### Back Propagation.
* This technique is used to train the neural network.
* In the ith iteration of training, the inputs are propagated through the model.
* The output is computed and compared with the actual output. The difference between 
the predicted and actual output(say ∆) is used to update the weights.
* The weights are updated backwards(output layer to input layer) using a learning rate
 and its derivative(gradients).
* These weights are then used for the (i+1)th generation. 