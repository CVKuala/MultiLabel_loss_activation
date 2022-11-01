# Multilabel_Activation

Let us tackle the problem of multi label classification by setting up a Neural Net that has the ability to classify multiple objects in an image at once. More specifically, the challenge we are tackling here is the RSNA 2022 Cervical Spine Fracture Detection being hosted by Kaggle. Since it would violate copyright issues if I were to publish data here, it is advised that you sign up for this competition to access the data, and then run any of the .ipynb files that are part of this repo. But before doing this, let us learn aboout the dataset.

The main problem we are solving through this project are -

1) Setting up a classifier capable of performing Multi Label Classification to classify fractures in CT Scans.
2) Finding out the optimal Activation Function + Loss combination for this challenge.

# A Little about the data

As mentioned earlier, we are going to tackle the RSNA 2022 Cervical Spine Fracture Detection challenge by using a Resnet model to perfrom multi-label classification. The goal of this challenge is to identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient. Quickly detecting and determining the location of any vertebral fractures is essential to prevent neurologic deterioration and paralysis after trauma. 

Each image in the dataset is in the dicom file format. The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Now some of the DICOM files are JPEG compressed. So some additional resources to read the pixel array of these files, such as GDCM and pylibjpeg will be required. These libraries are installed within the jupyter notebook files in the second cell itself.

![image](https://user-images.githubusercontent.com/90802245/194709195-acce878a-ea79-424e-a78f-698a5f1e4f2c.png)

This is how the train_data is structured, as you may have observed by now, there are 8 target columns with multiple positive predictions that can occur in any combination. We use a ResNet based classifier to solve this problem, however the question is what changes do we need to make to the architecture of ResNet to get this to work?

# Activation Functions and Loss Functions

A normal Multi-Class classifier would use a Softmax activation function, however, here the task is different. For multi-label problems, the labels are not mutually exclusive. This means that we can have multiple targets (labels) at the same time. The Softmax activation function will tend to favour one label at a time, so for example, if there are 4 correct labels out of 8 target ones, the softmax function will still only favour one. This will negatively affect the accuracy of such models.

Sigmoid on the other hand faces no such problem, the final layer of the ResNet will output values independent of each other meaning there is no bias towards any particular class.

Now, what about the loss function? It may be a well known fact that for multi-label classification, using Sigmoid + BCELoss is a more suitable combination than others. Alternatively, we could also have used BCEWithLogitsLoss() instead as this is the same as Sigmoid + BCELoss. However, to keep things simple we used the initial combination. However, there are contenders! The MultiLabelMarginLoss() has been specifically designed for multi-label classification. In this project we will test these different combinations out.

![image](https://user-images.githubusercontent.com/90802245/194718776-b56eb38a-9ebb-4f9f-b5b4-4fa38d8c2c44.png)

# How to run the Code

First lets clone our repository
```
https://github.com/bose1998/Multilabel_Activation.git
```
Now, lets install the requirements. Do make sure that before attempting to install all the requirements, you are in the required directory.
```
pip install requirements.txt
```
Now just execute the code.
```
python main.py --activation="Choose your activation function" --loss="choose your loss function" --lr=0.1 --epochs=20
```
![image](https://user-images.githubusercontent.com/90802245/194723566-305159ad-0651-4c71-b936-8afb1fd4defc.png)

A comparison of the training performance of the two methods (For BCELOSS)

![image](https://user-images.githubusercontent.com/90802245/194724029-360a8a53-c335-4774-becf-cf34561330f9.png)

Similar Comparison, except this time the perfromance on MultiLabelMarginLoss is being compared. Also note that the magnitude of loss for MultiLabelMarginLoss is lower than that for BCELoss. Further evaluation on Test set will prove that Sigmoid + MultiLabelMarginLoss is the best combination for this particular problem.

A similar Comparison has also been done for CrossEntropyLoss and MultiLabelSoftMarginLoss. Go through the code to evaluate its performance.

# References

I could not have completed this project without the following material -

https://github.com/kuangliu/pytorch-cifar

This particular repository was instrumental for this project as it taught me how to set up a ResNet model. I learned a lot about the architecture of such Neural Nets through this, and I would recommend that people go through this repo for better understanding of the architectures of various CNNs.

https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection

Do make sure to accept competition rules before trying to run the notebook files. I must also note that since the dataset was very large, I could not train the model on the entire dataset. However, feel free to change the amout of training data as you please. 
