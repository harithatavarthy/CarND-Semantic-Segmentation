# Project 12: CarND-Semantic-Segmentation-Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image_example]: ./runs/1524341126.2543495/um_000008.png
[image_Decoder]: ./images/SemanticSegmentation_Decoder.png
[image_loss]: ./images/Loss.png
[image_accuracy]: ./images/Accuracy.png
[image_gif]: ./images/Webp.net-gifmaker.gif


## Introduction
The goal of this project is to label the pixels of a road in the given testimages with about 80% accuracy, using a Fully Convolutional Network (FCN).  as described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer, and Trevor Darrell from UC Berkeley. The projects is based on [Udacity's starter project](https://github.com/darienmt/CarND-Semantic-Segmentation-P2).

A semantic segmentation  model is build on top of a VGG-16 image classifer to retain the spatial information and then upsample to the original image size. Once the model is build, it has been trained by tuning hyper parameters in such a way that the overall loss of the model is minimized. After model is trained to reasonably sufficient levels, inferences are made against the test data/images.

The following image shows exemplarily the result of the VGG-16 based FCN which has been trained to determine road (green) and non-road (not marked) areas.

![Road Expample][image_example]

## Fully Convolutional Network (FCN) Architecture

The Fully Convolutional Network (FCN) is based on a pre-rained VGG-16 image classification network. The VGG-16 network acts as a encoder. In order to implement the decoder, I extracted layer 3, 4 and 7 from the VGG-16 ([`main.py` line 81](https://github.com/harithatavarthy/CarND-Semantic-Segmentation/blob/master/main.py#L81)) network and implemented several upsampling and skip connections ([`main.py` line 116](https://github.com/harithatavarthy/CarND-Semantic-Segmentation/blob/master/main.py#L116)). The image below shows the schematic Decoder architecture. 

![Decoder architecture][image_Decoder]

Each convolution `conv2d()`and `conv2d_transpose()` of the decoder has been setup with a kernel initializer (`tf.truncated_normal_initializer`) and a kernel regularizer (`tf.contrib.layers.l2_regularizer`). This will ensure quick convergence of training loss as well as elimination of features that are least importance for learning.

## Training on AWS EC2 Instance
The FCN has been trained on an Amazon Web Services (AWS) EC2 g3.4xlarge instance with the following hardware configuration.

- 16 vCPUs (Intel Xeon E5-2670)
- 1 GPU (Nvidia Tesla M60)
- 7.5 GB RAM
- 100 GB SSD

I have created this EC2 using  community AMI provided by udacity-carnd-advanced-deep-learning (ami-effbaa94).
By using this AMI, one need not worry about complexity invovled in creating the environment required to train and run the model.


### Training Set
As training set I used the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). It consists of 289 training and 290 test images. Furhter details can be found in the [ITCS 2013 publication](http://www.cvlibs.net/publications/Fritsch2013ITSC.pdf) from Jannik Fritsch, Tobias Kühnl, Andreas Geiger.

| abbreviation | #train | #test | description                |
|:-------------|:------:|:-----:|:---------------------------|
| uu           |   98   |  100  | urban unmarked             |
| um           |   95   |  96   | urban marked               |
| umm          |   96   |  94   | urban multiple marked lane |
| URBAN        |  289   |  290  | total (all three subsets)  |

### Hyper-Parameters
 The total training time for 289 training samples took about 2 hours end to end using the below values for the hyper parameters

| Parameter           |  Value  |
|:--------------------|:-------:|
| KERNEL_INIT_STD_DEV |  0.01   |
| L2_REG              |  1e-3   |
| KEEP_PROB           |   0.5   |
| LEARNING_RATE       | 0.0009  |
| EPOCHS              |   50    |
| BATCH_SIZE          |    5    |


In order to arrive at the optimal model, i trained the model several times with multiple combinations of the above hyperparameters with varied values. Below are some of the observations.

a. With a bigger batch size, time taken for each iteration of training was less but the number of iterations required to arrive at high levels of accuracy was more

b. Smaller batch size resulted in iterations with increased divergence in training loss making it unreliable.

c. Higher values for L2_REG resulted in increased penalization of features resulting in losing valuable information

d. Lower values for L2_REG resulted in noise

e. Very low values for learning rate increased the overall training time


The two diagrams below depict the decrease in cross entropy loss over time and increase in accuracy over time.

![loss][image_loss] ![accuracy][image_accuracy]

### Tensorboard
In order to visualize the FCN graph, the cross entropy, Accuracy and intersection-over-union progress I added several summary operations into the model and training methods. All relevant TensorBoard log files are stored in the `tensorboard_log` directory. The TensorBoard can be started with the following command. 

```
tensorboard --logdir=tensorboard_log/
```

## Results
The FCN classifies the road area quite well. It has some trouble to determine the road area in rail sections or in scenes with heavy cast shadows. This is due to the fact, that the training set with only 289 images is very small and not all scenarios in the test set are covered by the training set. By applying bigger training sets or image augmentation the performance could be further improved.

![Results GIF][image_gif]


#--------------------------------------------------------------------------------------------------------------
#UDACITY README

# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
