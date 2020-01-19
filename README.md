# CourseFineGrainedClassification

Given a set of images, our task is to identify the objects belonging to classes :aeroplane, bottle and chair, and draw a bounding box around it.

## Methods used

### Coarsed grained classfication

####  Pretrained VGG

* We first tried a pretrained VGG-16 model, freezing the pretrained layer,and replacing the FC layers with only 1 FC layer of (25088 x 5)), whichlead  to  exponential  decrease  in  trainable  parameters,  and  thus  trainingwas a lot faster.

* It gave decent results on course grained classfication,  but we fine tunedall the layers on our dataset in order to learn better weights incorporatingfeatures with respect to our images.

* We  obtained  the  pretrained  model  of  VGG-16  on  Imagenet  data  fromtorchvision module.

### Results
We observed that combining all the images of all classes, our model was able torecognize the correct coarse grained class 98-99% times on the test set that wekept - 10%


### Fine-grained classfication

#### Bilinear-CNN

![](https://github.com/prerit2010/CoarseFineGrainedClassification/blob/master/diagram.png)

* We used our VGG classifier fine tuned in the previous step as the basemodel for Bilinear-CNN.

* We froze all the layers of this pretrained model trained on coarse grainedclasses, and added just 1 FC layer to do fine grained classfication for eachcoarse grained class.
