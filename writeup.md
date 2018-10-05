# roboND Proj4: Follow Me
### Submission writeup: ChrisL 2018-10-04<br/>
**Notes to Reviewer:**<br/>
The final h5 training file is [./data/weights/model_weights](./data/weights/model_weights)<br/>
The notebook can be viewed as
 [model_trainingNB.html](./code/model_trainingNB.html),  or  [model_training.ipynb](./code/model_training.ipynb),
---

## **Project Overview**
The goals / steps of this project are the following:
Using Keras/tensorflow in a Jupyter/python notebook environment:

* Develop a Fully Convolution Neural network (FCN) model for semantic image segmentation using Keras
    * Define the convolutional encoder layers
    * Define the corresponding convolutional decoder layers
    * Integrate them in an FCN model
    
* Train and tune the FCN model using the provided training images and corresponding label masks
    * Iteratively train the model and test against the provided validation images and masks
    * Modify the hyper parameters and/or the model to improve accuracy
    * \[Optional\] Use the sim to generate more training images to use for improved training accuracy
    
* Use the model guide a drone in a simulator to follow the target
    * Start the drone sim and the sim control client ./code/follower.py 
    The sim client is provided and takes the trained model as input and uses 
    it to segment the image stream from the drones camera and uses
    the identified target pixel locations to generate pose commands
    to send to the sim in order to stay near to the target. 

    
---


## Model Overview
My model is an FCN consisting of 3 layers of encoding/decoding convolutions, with a single 1x1 convolution
in the middle, followed by a final output convolution classification layer with softmax activation.
  
Training is performed using the provided Adam optimizer operating on the cross entropy loss function. The training
runs a variable number of epochs on the training data set that is split into batches of a variable size and using
a variable training rate.

Final scoring used Intersection over Union analysis.

### Model Details
Here is a graphical representation of my model, with discussion below.

![modelGraph][refImage_ModelGraph]


#### Encoders
Each encoder layer consisted of a separable_conv2d with a stride of 2 and
varying numFilter/output depths and a batchnorm and relu activation. As the training progresses
the different encoder filters will become responsible for recognizing features in a way that is translatable
to any location on the input tensor, ie a given filter will activate for a given learned feature type 
at any/all locations of the input as it is swept across. Since each filter is 're-used' like this
it is equivalent to sharing the parameters of that filter and therefore cuts down on the total number of
parameters for the network.

I did not play much with the number of filters at each encoder level. I retrospect
I think I have the filter counts, 32, 64, & 128 backwards from what they should be. 
It is my impression now that the first encoder convolutional filters are activating
for very primitive features, for example a diagonal segment, 
and that there very few features to be activated on the last encoder.

```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def encoder_block(input_layer, filters, strides):    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)    
    return output_layer
```

 
Each encoder layer results in a downsampled feature layer.

#### Middle 1x1 Convolution layer
The 1x1 convolution layer connects the encoder layer sequence with the decoder layer sequence.
It is essential for accumulating spatial feature information into a deep tensor that is otherwise lost 
for example in a fully connected layer output. 

A fully connected output network is OK for creating a
final classification prediction, but it retains no information about which pixels were responsible for
that classification. The 1x1 convolution retains that information and the decoding layers
are responsible for rebuilding an array of classified pixels the same size as the original.

```
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
    
...
# TODO Add 1x1 Convolution layer using conv2d_batchnorm().
conv1x1 = conv2d_batchnorm(enc3, filters=64, kernel_size=1, strides=1)
...


```

#### Decoders
Each decoder layer consisted of a bilinearUpsample2D, a 'skip connection'
concatenation of the higher resolution encoder layers and 2X SeparableConv2D+batchNorms.
The decoding layers are responsible for rebuilding an array of classified pixels the same size as the original
by upsampling and convolution. 
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
    
def decoder_block(small_ip_layer, large_ip_layer, filters):    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    combined = layers.concatenate([upsampled, large_ip_layer])
    # TODO Add some number of separable convolution layers
    sep1 = separable_conv2d_batchnorm(combined, filters)
    output_layer = separable_conv2d_batchnorm(sep1, filters)
    
    return output_layer
```


#### The FCN model
Here is the implementation of the FCN that assembles the components above, the encoders, 1x1Convolution, and decoders
and final output layer.

**FCN pros and cons**<br/>
The FCN model we are using is good for our semantic segmentation needs of classifying each pixel of the input image. 
And it would be be fine to recognize different kinds of objects, such as cat from dog from car, with suitable
training inputs. However it should be noted that the follow-me target had a very uni1que appearance (all red, eg) compared to
the other sim humans. For this FCN to work with different kinds of objects it would require that the target be
visually unique (enough) and that there be only one instance of the target present. Also the training data labels are provided
as single channel masks packaged in an RGB image, a mechanism that only permits 3 classes. For more classes a different
way of delivering the masks into the FCN would be required.

Also as is this FCN is not adequate for distinguishing multiple instances of classified objects. 
That would require further processing or a different kind
of Neural Network

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc1 = encoder_block(inputs, filters=32, strides=2)
    enc2 = encoder_block(enc1, filters=64, strides=2)
    enc3 = encoder_block(enc2, filters=128, strides=2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1x1 = conv2d_batchnorm(enc3, filters=64, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec1 = decoder_block(conv1x1, enc2, filters=128)
    dec2 = decoder_block(dec1, enc1, filters=64)
    dec3 = decoder_block(dec2, inputs, filters=32)
        
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec3)
```

#### Training, hyperparameters and tuning

Here are the final hyperparameters, detailed below, that I used to generate the final output model. 
The tuning process, detailed below, was laborious and time consuming (about 90 minutes for final training)
and was influenced by constraints of my computer.

```
learning_rate = 0.001
batch_size = 32
num_epochs = 30
steps_per_epoch = 200
validation_steps = 20
workers = 2
```

**Num Epochs**<br/>
This is the number of training passes that are run through the model training process. I found that 10 passes
did not yield adequate accuracy. 20 was tolerably close but due to the long training times in comparison
to the time I had for optimizing I hedged and went with 30 and found that it generated a a good result.
It is possible that this resulted in overfit and with faster hardware or more time I would have investigated optimizing.

**Batch Size**<br/>
For shorter training time it is desirable to have a larger batch size. However I found that a batch size of 64
immediately failed with 'OOM' out of memory errors when I started training. Even worse were size 40 and 50
which failed at some time well into training and seemed dependant on other activities on the computer
that put demands on the GPU memory. I settled on a _batch_size = 32_ that was able to reliably complete training runs.

**Steps per epoch**
A better name for this parameter is number of batches per epoch. 
I chose this value to ensure that, at minimum, all of the available training images would be used in every epoch. 
In the final model the number of images run exceeded the number of available images (4131) so that there were repeats. 
While this does no harm if the images selected are shuffled it could have been time optimized as 
steps_per_epoch = num_images/batch_size 
to ensure that every training image is run exactly once.

**Learning Rate**<br/>
I tried several learning rates. Higher rates improve the accuracy faster initially but soon plateau. 0.001 Provided
rapid enough training that seemed to improve steadily through all 30 epochs.

**Validation Steps**
I chose this low number for faster training but in retrospect I should have left it much higher as the validation
loss plots were very erratic, I believe because the small sample size meant that the average
 loss value was therefore very noisy.
 
**Workers**<br/>
Left as default. I think this is number of CPU threads and the training generally saturated and was was limited
primarily by GPU throughput.

### Future Enhancements
**Encoder/Decoder numFilters**<br/>
As noted above I did not much experiment with these parameters and I suspect that the parameters
I have could be much more efficient or effective.

**More parameter fiddling**<br/>
I did not have enough time and a fast enough computer to really explore the tuning options. 
An automated system to try combinations of widely varied parameters on a faster system could probably 
improve the accuracy of this system.

**More training data**<br/>
Since the accuracy reached 0.42 I did not feel a need to generate more training data, 
but as always more training will probably improve the model.

**Dropouts**<br/>
I found random layer drop outs to be very helpful in other NN and suspect it could help here.


---


## Links
[ProjectRubric][refRubric]<br/>
[My Project Repo][refProjectRepoMine]<br/>
This, my main project repo.

[Udacity Main Project Template repo][refProjectRepoUda]<br/>
[Udacity Proj Exercise1 Repo][refExercise1RepoUda]<br/>
[Udacity Proj Exercise2 Repo][refExercise2RepoUda]<br/>
[Udacity Proj Exercise3 Repo][refExercise3RepoUda]<br/>


[//]: # (WWW References)
[refRubric]: https://review.udacity.com/#!/rubrics/1155/view
[refProjectRepoUda]: https://github.com/udacity/RoboND-DeepLearning-Project
[refProjectRepoMine]: https://github.com/cielsys/RoboNDProj4_FollowMe

[refExercise1RepoUda]: https://github.com/udacity/RoboND-NN-Lab/
[refExercise2RepoUda]: https://github.com/udacity/RoboND-CNN-Lab
[refExercise3RepoUda]: https://github.com/udacity/RoboND-Segmentation-Lab

[trainingsetdownload]: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
[projectinstructions]: https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/8cb6867c-f809-49b3-9bc1-afb409a112a7
[GTSRB]: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

[//]: # (Image References)
[//]: # (html resizeable image tag <img src='./Assets/examples/placeholder.png' width="480" alt="Combined Image" />)

[refLedeImage]: ./Assets/finalTest/01.jpg
[refImage_1]: ./Assets/finalTest/01.jpg
[refImage_ModelTextTable]: ./docs/writeupImages/model_summary.txt
[refImage_ModelGraph]: ./docs/writeupImages/model_plot.png
[refImage_ModelGraphSimple]: ./docs/writeupImages/model_plotsimple.png


[trainhist]:  ./Assets/writeupImages/trainhist.png     
[trainhistaug]:  ./Assets/writeupImages/trainhistaug.png     
[extraimages]: ./Assets/writeupImages/extraimages.png     
[trainsample]:  ./Assets/writeupImages/trainsample.png     
[02zoomed]:  ./Assets/writeupImages/02zoomed.png
[augmentssmall]:  ./Assets/writeupImages/SampleAugmentsSmall.png
[augmentsbig]:  ./Assets/writeupImages/SampleAugmentsBig.png
[extrapredictions]:  ./Assets/writeupImages/predictions.png
