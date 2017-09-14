# Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

### Required files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

### Quality of code

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

#### An appropriate model architecture has been employed

I have used the NVIDIA architecture as my neural network model. It has the following layers:

* Lambda layer to normalize the input values between -0.5 and 0.5
* Cropping layer to get rid of the bottom and top part of the image to focus more on the road
* 5 Convolutional layer with valid padding and relu activations
* Flattening layer
* 5 dense layers, 4 using relu activationa and last one using tanh activation

Here is the model summary from Keras

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          2459532     flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================
Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0
```


#### Attempts to reduce overfitting in the model

I have made a few attemps to use dropout in my model. However the results turn out to be worse than without the dropout.
My understanding is that dropout could help to generalize if I had more training data. In the end I was able to train the
model to drive on track 1 or track 2 without using dropout.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The only thing I tuned after switching to nvidia architecture was to increase image cropping from top. Rest was playing with the input data.

#### Appropriate training data

To gather the training data I followed the suggestions in lectures: two laps of smooth driving, one lap going from side to side and one lap focused on doing the curves as smooth as possible. I have used mouse to drive the car. More details are in the following section.

###Model Architecture and Training Strategy

#### Solution Design Approach

**NOTE:** Videos and model.py from all runs are included in folders run3 to run19. Model files are added to github only for the successful runs.

##### Run 1 and 2
I have started with a little bit of data generated by myself driving the car using keyboard. First two runs where purely to figure out if both training and driving work fine for me. It was a simple two layer feed forward network. Not a big surprise the car went offroad pretty quickly. However I got the confirmation that the workflow works fine and there are not technical difficulties.

##### Run 3 and 4
As we process images I decided to give LeNet a try. The result was better as car at least stayed on the road for a moment. First run was on my limited data and second on udacity provided data. With udacity data the car was able to drive a little bit further. At that moment I was already cropping the image 20 pixels from the bottom and 50 from the top.

[Video run 3](run3/run3.mp4)
[Video run 4](run4/run4.mp4)

##### Run 5
For the run 5 I switched into NVIDIA architecture still using the udacity provided data. The car almost made it through the corner after the bridge. It raised my hopes and was matching the tips on slack channel: use nvidia, it works.

[Video run 5](run5/run5.mp4)

##### Run 6
As a try to improve the situation I decided to crop the image 75 pixel from the top instead of just 50 and also extend my set with flipped images. To my surprise car completed the lap for the first time. However the driving was not the best, going on to the lines a few times. Also the results were not consistent, car sometimes fell out of the road.

[Video run 6](run6/run6.mp4)

##### Run 7 and 8
In a hope for better corners I decided to extend the training data with side cameras. Run 7 included all three cameras without flipping and run 8 was extended with flipped versions of all three cameras. The correction paramter used for the side cameras was 0.2. For run 7 I have used the correction parameter incorrectly for right camera so the car keeps on the side of the road. Run 8 result was not satisfactory as well, the car went offroad again.

[Video run 7](run7/run7.mp4)
[Video run 8](run8/run8.mp4)

##### Run 9 and 10
Previous runs made me think that the provided data may not be good and may not have enough variation in recovering from the corners. I decided to record my own data again, this time using mouse as a controller. I kept the nvidia architecture and reduced the input data just to center camera with flipping. Not a great result. At this moment I was not sure about data quality. I decided to try again without any architecture changes but with all 3 cameras. It gave a better result.

[Video run 9](run9/run9.mp4)
[Video run 10](run10/run10.mp4)

##### Run 11 - COMPLETED LAP
I remembered that the correction parameter for the side cameras should be tuned as well. I decided to try with smaller correction and see how it affects the driving. It turned out to be the key to fix the current model. I have achived a model that can drive the car around the whole track. The longest try was about 25 minutes (and then I stopped it, it didn't crash).

[Video run 11](run11/run11.mp4)

##### Run 12 to 14
During these runs I tried to use dropout at various places of the model: after the convolutional layer and after the dense layers. However the results were degrading. As the original nvidia architecture did not use dropouts I decided that three tries is enough especially that I already had a model running a full lap.

[Video run 12](run12/run12.mp4)
[Video run 13](run13/run13.mp4)
[Video run 14](run14/run14.mp4)

##### Run 15 and 16
Run 15 and 16 were the same runs as run 11 but I have changed the speed in drive.py. With speed set to 20 car could drive a lap. With speed set to 30 car completed the lap but the driving was a little bit crazy. My intuition is that with higher speeds the difference between subsequent frames fed into the network are bigger and less smooth and that causes the predicted steering angle to be too drastic to drive smoothly. I assume that with more training data we could make it a little bit better.

[Video run 15](run15/run15.mp4)
[Video run 16](run16/run16.mp4)

##### Run 17 to 19, track 2
I decided to try architecture from run 11 on track 2. I have gathered training data from track 2 using mouse as a controller, two laps. The results were not bad but the car crashed. I gather the data again, three laps: two smooth and one from side to side. The effects were better but still car crashed from time to time. Then I combined both data recording and trained the model again. This time it could complete the lap on track 2.

[Video run 17](run17/run17.mp4)
[Video run 18](run18/run18.mp4)
[Video run 19](run19/run19.mp4)


#### Final Model Architecture

The final model architecture stayed as per nvidia whitepaper and is described above.

#### Creation of the Training Set & Training Process

Creation of training set and training process is described above

## Final thoughts

After doing several runs it is pretty clear that more than a good architecture it is important to have a quality data. With projects like this I realize that "quality" is very hard to define. My initial data was gathered by driving the simulator using keyboard without leaving the track. However that was not enough to allow the model to generalize well and especially to learn how to recover. It turned out that I spent more time on playing with the data then with the network architecture itself.