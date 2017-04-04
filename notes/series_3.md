# Lecture Series III (Convolutional Neural Nets)

## Intuition
* Similar objects(images/text) share the same weights irrespective of their location.
* This is called translational invariance.

## Convnets
* Share parameters across space.
* Assume n image has a width, height and depth(RGB).
* A small patch/kernel is moved across it such that the width and height is reduced and the depth is increased/decreased.
* This can be repeated multiple times with multiple kernels where the patch size and weights for each layer is changed.
* Eventually, the width and height of the image is squeezed and the depth becomes the class contribution. 
* The movement of the patch is denoted as its stride. Larger the stride, greater the image shrinks.
* Once the image size is reduced after a few layers of convolution, a fully connected layer can be used do make the classifier


## Improvements
### Pooling
* Strides are a very aggressive form of downsampling an image.
* So instead of doing a convolution, a patch of image is taken and all values are replaced by an exemplar value.
* If the exemplar value is replaced by the max value in the patch, it is called max-pooling.
* If the exemplar is the mean value, it is called average pooling.

### 1x1 Convolutions
* Esentially a matrix multiplication
* Used to make the network deeper with lesser number of parameters.
* Structure of the image(width, height) does not change much.

### Inception
* It is a combination of Pooling/Convolution of varying sizes and eventually the result of the combination is concatenated.