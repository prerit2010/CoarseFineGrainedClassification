# CoarseFineGrainedClassification

Given  a  dataset  which  has  some  coarse  grained  classes,  and  within  each  coarsed grained  class,  we  have  some  fine  grained  classes. The aim is to test the  accuracy  of  an  algorithm  on  the  task  of  coarse  and  fine  grained classification

## Methods used

### Coarsed grained classfication

####  Pretrained VGG

* Tried a pretrained VGG-16 model, freezing the pretrained layer, and replacing the FC layers with only 1 FC layer of (25088 x 5)), which lead to exponential  decrease in trainable parameters,  and thus  training was a lot faster.

* It gave decent results on coarse grained classfication, but we fine tuned all the layers on our dataset in order to learn better weights incorporating features with respect to our images.

* We  obtained  the  pretrained  model  of  VGG-16  on  Imagenet  data  from torchvision module.

### Results

Observed that combining all the images of all classes, the model was able to recognize the correct coarse grained class 98-99% times on the test set that we kept - 10%


### Fine-grained classfication

#### Bilinear-CNN

code credit : https://github.com/HaoMood/bilinear-cnn

![](https://github.com/prerit2010/CoarseFineGrainedClassification/blob/master/diagram.png)

* Used our VGG classifier fine tuned in the previous step as the basemodel for Bilinear-CNN.

* Froze all the layers of this pretrained model trained on coarse grainedclasses, and added just 1 FC layer to do fine grained classfication for each coarse grained class.
