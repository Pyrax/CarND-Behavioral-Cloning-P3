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
[model]: ./output_images/model.png "Visualization of model architecture"
[model_loss]: ./output_images/model_loss.jpg "Graph of training and validation loss"
[offset_comparison_hist]: ./output_images/offset_comparison_hist.jpg "Distribution of steering angles for different offsets"
[roi]: ./output_images/roi.jpg "Region of interest"

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

First of all, I collected driving data by recording runs of both tracks of the Udacity Simulator in forward and reverse 
direction. I additionally recorded recovery driving from the sides of the road on each track. This helps the model to 
recover itself when the car steers off center. 

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
The normalization code can be found here: [udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim/blob/bdcd588990df2db4706e772cd45d6e013631a2f2/Assets/Standard%20Assets/Vehicles/Car/Scripts/CarController.cs#L472). 
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

### Model architecture & training

![model]

### Result

### Discussion
