# Project: Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[augment_brightness]: ./output_images/augment_brightness.jpg "Brightness augmentation"
[augment_noise]: ./output_images/augment_noise.jpg "Salt & pepper noise augmentation"
[augment_rotation]: ./output_images/augment_rotation.jpg "Image rotation"
[center_steering_hist]: ./output_images/center_steering_hist.jpg "Distribution of steering angles of middle camera"
[example_image]: ./output_images/example_image.jpg "Example of image data featuring all cameras"
[feed_images]: ./output_images/feed_images.jpg "Images fed into the network"
[full_steering_hist]: ./output_images/full_steering_hist.jpg "Distribution of steering angles on final data set"
[model_loss]: ./output_images/model_loss.jpg "Graph of training and validation loss"
[offset_comparison_hist]: ./output_images/offset_comparison_hist.jpg "Distribution of steering angles for different offsets"
[roi]: ./output_images/roi.jpg "Region of interest"

[//]: # (References)

[1]: https://github.com/udacity/self-driving-car-sim/blob/bdcd588990df2db4706e772cd45d6e013631a2f2/Assets/Standard%20Assets/Vehicles/Car/Scripts/CarController.cs#L472
[2]: http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf
[3]: https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
[4]: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
[5]: https://stackoverflow.com/a/30624520
[6]: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

Details About Files In This Directory
---

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

Writeup
---

### Data collection & pre-processing

First of all, I collected driving data by recording runs of center driving of both tracks of the Udacity Simulator in 
forward and reverse direction. I additionally recorded recovery driving from the sides of the road on each track. 
This helps the model to recover itself when the car steers off center. 

The following figure presents images of all three different cameras of an example frame:

![example_image]

The recorded data set has the following characteristics:

description | value
------------|--------
samples per camera | 9742
number of cameras | 3
total samples | 29226
min steering ratio | -1.0
max steering ratio | 1.0
mean steering ratio | -0.0011161202688359676

Data shows that steering angles in the driving log are already normalized to be in range between -1 and 1. For 
demonstration purpose I also want to look at the real angles which means that angles have to be scaled back to degrees. 
The normalization code can be found here: [udacity/self-driving-car-sim][1]. 
So, all angles are divided by the maximum steering angle by the simulator before writing to driving_log which equals 
25° in the current version. 

Here, the distribution of steering angles only for the center camera is illustrated:

![center_steering_hist]

In order to take advantage of the side cameras as well, I examined how histograms look like if I insert the images of 
the side cameras as additional samples with a fixed offset:

![offset_comparison_hist]

For the final architecture I have used an offset of 0.25 as it shows a rather balanced distribution and it empirically 
performed best while evaluating the different offsets on the same model during simulator tests.

Furthermore, I added a copy of each image with inverted steering angle for flipping later which has the benefit of 
doubling the amount of data and normalizing data to zero mean. It leads to this new chart:

![full_steering_hist]

Moreover, a region of interest has been defined to focus on important road features and to exclude the horizon and car's 
bonnet as demonstrated below:

![roi]

### Data augmentation for training

As described in ["The Effectiveness of Data Augmentation in Image Classification using Deep Learning"][2] even simple 
traditional data augmentation methods like flipping, rotation or distortion can increase validation accuracy 
significantly with a reasonable level of effort and additional computation cost. Also, with the so-called 
["augmentation on the fly"][3] (augmentation during batch generation) we can in theory generate an infinite amount of 
data although our original data set is limited. This can help to reduce overfitting. 

Therefore, I implemented a few augmentation techniques that are fairly simple to realize for our project's 
domain. Those I have used in this project are: image flipping, brightness adjustment, rotation, and noise.
These are less complicated as opposed to e.g. image shifting which requires special attention as it might change 
meaning of our data because it modifies road features. Then, steering angles would have to be adjusted.

The first method is flipping of the images that is done during batch generation where the actual image data is loaded. 
The data frame already contains two rows for each image filename as described above. So, only the image manipulation 
still has to be performed at generation.

Next, the samples are randomly adjusted in terms of brightness using gamma correction as described in ["Changing the 
contrast and brightness of an image!"][4]. For reduced computation costs, as data augmentation quickly becomes a 
performance bottleneck, I have pre-computed some lookup tables for a range of different gamma values:

![augment_brightness]

Secondly, a random amount of salt & pepper is applied as noise as described on [StackOverflow][5]. It is (again) 
designed for fast execution:

![augment_noise]

Last but not least, images are rotated by a range between `-10.0°` and `+10.0°`:

![augment_rotation]

### Model architecture & training

My model architecture is based on [NVIDIA's PilotNet model][6]. However, my initial input size for images that are fed 
into the neural net is 160x320x3 and then gets cropped and resized to 66x200x3 images which is the input layer size in 
the paper. Then, the image is normalized by scaling the pixels by `n/127.5 - 1.0`. Normalization layer is then followed 
by 5 convolutional layers, 3 fully-connected layers and 1 output layer. As the NVIDIA paper does not specify the type of 
activation functions used I decided to use ReLUs for every layer except the output layer for which I have used `tanh`. 
Alternatively, I could have also left the activation function out after the output layer to solve our linear regression 
problem but through hyperbolic tangent I achieved smoother steering, although loss is similar. This improvement might 
come from `tanh` operating in the same interval between `-1.0` and `1.0` as our steering data which enables smooth 
transitions.

After each activation I have also inserted batch normalization resulting in accelerated training through faster 
convergence and slightly reduced overfitting.

This finally leads to the following model:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_image (InputLayer)     (None, 160, 320, 3)       0         
_________________________________________________________________
image_cropping (Cropping2D)  (None, 70, 260, 3)        0         
_________________________________________________________________
image_resize (ResizeImages)  (None, 66, 200, 3)        0         
_________________________________________________________________
image_normalization (Lambda) (None, 66, 200, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 31, 98, 24)        1824      
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 98, 24)        96        
_________________________________________________________________
conv2 (Conv2D)               (None, 14, 47, 36)        21636     
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 47, 36)        144       
_________________________________________________________________
conv3 (Conv2D)               (None, 5, 22, 48)         43248     
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 22, 48)         192       
_________________________________________________________________
conv4 (Conv2D)               (None, 3, 20, 64)         27712     
_________________________________________________________________
batch_normalization_4 (Batch (None, 3, 20, 64)         256       
_________________________________________________________________
conv5 (Conv2D)               (None, 1, 18, 64)         36928     
_________________________________________________________________
batch_normalization_5 (Batch (None, 1, 18, 64)         256       
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
dense1 (Dense)               (None, 100)               115300    
_________________________________________________________________
batch_normalization_6 (Batch (None, 100)               400       
_________________________________________________________________
dense2 (Dense)               (None, 50)                5050      
_________________________________________________________________
batch_normalization_7 (Batch (None, 50)                200       
_________________________________________________________________
dense3 (Dense)               (None, 10)                510       
_________________________________________________________________
batch_normalization_8 (Batch (None, 10)                40        
_________________________________________________________________
output_angle (Dense)         (None, 1)                 11        
=================================================================
Total params: 253,803
Trainable params: 253,011
Non-trainable params: 792
_________________________________________________________________
```
(another diagram describing the model can be found at [output_images/model.png](./output_images/model.png))

As the total size of our data set is larger I have used a generator to create batches for training and validating the 
network. The following figure shows a few samples that were fed into the network using the generator:

![feed_images]

Training the model is then performed using the Nadam-optimizer and mean squared error-loss. For the optimizer I had to 
reduce the learning rate from the default value of 0.002 to 0.001. Otherwise, the model converged at higher loss and 
testing the model sometimes even led to the model only predicting one value for any input data.

Moreover, I tested to include some dropout layers in my model architecture. First, I tried to only add dropout after the 
first fully-connected layer with a dropout probability of 0.2. Secondly, I also experimented with dropout after each 
fully-connected layer with a probability of 0.2 on the first attempt and 0.5 and the second attempt. But, all these 
efforts resulted in worse loss and driving performance. Hence, I did not include dropout in my final neural network.

Lastly, I have evaluated the behavior of different color spaces on my model. The NVIDIA paper proposes to use YUV images 
what caused my car to drive on the yellow lane markings instead of driving in the middle of the road. HSV and Lab also 
caused the vehicle to steer off track. In the end, I processed images in RGB format which yields desired results.

### Result

Training the model for a total of 20 epochs accomplished the following MSE loss curve:

![model_loss]

My model is able to steer the car on itself for several rounds on both tracks and if it is manually steered away from 
center it will find its way back to the center.
The following two videos demonstrate the result on each track:
- [First track](./result/first_track/output_video.mp4)
- [Second track](./result/second_track/output_video.mp4)

### Further work

To improve the model further, visualization techniques like filter activation images and saliency maps could be 
explored to see which features the model picks up best and where attention could still be improved. 

Also, data augmentation could be expanded with more techniques to generate even more samples as the previously deployed 
methods have already proven to enhance the neural network's ability to learn.
