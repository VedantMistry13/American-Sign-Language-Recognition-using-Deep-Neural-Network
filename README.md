# American Sign Language Recognition using Deep Neural Network (Transfer Learning Approach)

## 1 - Introduction
American Sign Language (ASL) is a complete, natural language that is expressed using the movement of hands and face. ASL provides the deaf community a way to interact within the community itself as well as to the outside world. However, not everyone knows about signs and gestures used in the sign language. With the advent of [Artificial Neural Networks](https://medium.com/technology-invention-and-more/everything-you-need-to-know-about-artificial-neural-networks-57fac18245a1) and [Deep Learning](https://www.mathworks.com/discovery/deep-learning.html), it is now possible to build a system that can recognize objects or even objects of various categories (like red vs green apple). Utilizing this, here we have an application that uses a deep learning model trained on the ASL Dataset to predict the sign from the sign language given an input image or frame from a video feed. You can learn more about the American Sign Language over [here](https://www.nidcd.nih.gov/health/american-sign-language) [National Institute on Deafness and Other Communication Disorders (NIDCD) website].

#### Alphabet signs in American Sign Language are shown below:
![American Sign Language - Signs](/images/NIDCD-ASL-hands-2014.jpg)

## 2 - Approach
We will utilize a method called [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) along with [Data Augmentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b) to create a deep learning model for the ASL dataset.

### 2.1 - Dataset
The network was trained on this kaggle dataset of [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet). The dataset contains `87,000` images which are 200x200 pixels, divided into `29` classes (`26` English Alphabets and `3` additional signs of SPACE, DELETE and NOTHING). 

### 2.2 - Data Augmentation
So, as to train the model for better real-world scenarios, we have augmented the data using brightness shift (ranging in `20%` darker lighting conditions) and zoom shift (zooming out up to `120%`).

### 2.3 - Transfer Learning (Inception v3 as base model)
The network uses Google's [Inception v3](https://arxiv.org/pdf/1512.00567.pdf) as the base model. The first `248` (out of `311`) layers of the model (i.e. up to the third last inception block) are locked, leaving only the last 2 inception blocks for training and also remove the Fully Connected layers at the top of Inception network. We then create our own set of Fully Connected layers and add it after the inception network so as to conform the neural network for our application (consists of `2` Fully Connected layers, one consisting of `1024` ReLu units and the other of `29` Softmax units for the prediction of `29` classes). The model is then trained on the set of new images for the ASL Application.

### 2.4 - Using the model for the application
After the model is trained, it is then loaded in the application. [OpenCV](https://opencv.org/) is used to capture frames from a video feed. The application provides an area (inside the green rectangle) where the signs are to be presented to be detected or recognized. The signs are then captured in frames, the frame is processed for the model and then fed to the model. Based on the sign made, the model predicts the sign captured. If the model predicts a sign with a confidence greater than `20%`, the prediction is presented to the user (`LOW` confidence sign predictions are predictions above `20%` to `50%` confidence which are presented with a `Maybe [sign] - [confidence]` output and `HIGH` confidence sign predictions are above `50%` confidence and presented with a `[sign] - [confidence]` output where `[sign]` is the model predicted sign and `[confidence]` is the model's confidence for that prediction). Else, the model displays `nothing` as output.

**Note: You can download the notebook (American_Sign_Language_Recognition.ipynb) or the PDF version of the notebook (American Sign Language Recognition.ipynb - Colaboratory.pdf) to have a better understanding of the implementation.**

## 3 - Results
For training, [Categorical Crossentropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) was used to measure the loss along with [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) optimizer (with `learning rate` of `0.0001` and `momentum` of `0.9`) to optimize our model. The model is trained for `24` epochs. The results are displayed below:

### 3.1 - Tabular Results
Metric | Value
-------|------
Training Accuracy | 0.9887 (~98.87%)
Training Loss | 0.1100
Validation Accuracy | 0.9575 (~95.75%)
Validation Loss | 0.1926
Test Accuracy | 96.43%

### 3.2 - Graphical Results
![Training vs Validation Accuracy](/images/train_vs_val_acc.png)
![Training vs Validation Loss](/images/train_vs_val_loss.png)

## 4 - Running the application
If you want to try out the application, you might have to satisfy some requirements to be able to run it on your PC.

### 4.1 - Requirements
- [Python](https://www.python.org/downloads/) v3.7.4 or higher (should work with v3.5.2 and above as well)
- [NumPy](https://www.scipy.org/install.html)
- [OpenCV](https://solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/) v3 or higher
- [Tensorflow](https://www.tensorflow.org/install) v1.15.0-rc3 (may work on higher versions)[GPU version preferred]
- Might require a PC with NVIDIA GPU (at least 2GB graphics memory)
- Webcam

### 4.2 - Clone this repository
- Clone this repository using `git clone https://github.com/VedantMistry13/American-Sign-Language-Recognition-using-Deep-Learning`.

### 4.3 - Executing the script
1. Open a command prompt inside the cloned repository folder or just open a command prompt and navigate to the cloned directory.
1. Execute this command: `python asl_alphabet_application.py`.
1. An application with a window like the one shown below should pop up after some seconds or minutes (depends on the PC):
![Application Preview](/images/application_view.png)
1. Present the signs inside the green rectangular area provided by the application.
1. Watch the predictions along with the confidence score below the green rectangle.

## 5 - Outputs
Here are some of the outputs generated by the application in various lighting conditions:
![Preview 1](/images/preview_collection_1.png)
![Preview 2](/images/preview_collection_2.png)
![Preview 3](/images/preview_collection_3.png)
