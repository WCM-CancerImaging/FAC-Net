# FAC-Net
Fatty Acid Composition - Deep Neural Network
This code works as optimizer for estimation of water, Fat, Ndb, Nmidb, CL, R2*, Frq in MRI signal equation. 
This code consists of two parts one is data and other is Network part. Data code have input image of phantom MRI data of shape [256,256,16] with one slice and 16 echoes. It has two image data one is for magnitude part of complex MRI data and other for phase. FAC-Net has two U-nets one for magnitude part and other for Phase part of complex MRI data. Each U-Nets encoding and decoding path consist of 5 repeated blocks with a 2x2 convolution, sigmoid activation function in all layers except the last layer which have Linear Activation function, He Uniform as kernel initializer with no seed, batch normalization and a 2x2 max-pooling. Weights are initialized randomly. Training was conducted using the Adam optimizer with 7000 epochs and learning rate=0.0008. The magnitude part of FAC-Net predicted magnitude part of water, fat, Ndb, Nmidb and R2* while phase part of FAC-Net predicted phase part of water, fat, Ndb, Nmidb and frequency. 

Ndb and Nmidb are used to estimate MUFA, PUFA and SFA using the relation between Ndb, Nmidb, MUFA, PUFA and SFA.
Phantom data has 8 oils: [Peanut, Safflower, Coconut, Sunflower, Grapeseed, Canola, Flaxseed, Olive]. 

Download the input data: https://wcm.box.com/s/gfys45ayqwqpz0mqh8rojbfwx9jkqhd0

Download all the libraries listed in main.py and run.
This is the official repository of FAC-Net.

To be updated soon with python codes

## Reference
Chaudhary S, Lane EG, Levy A, McGrath A, Mema E, Reichmann M, Dodelzon K, Simon K, Chang E, Nickel MD, Moy L, Drotman M, Kim SG, Estimation of fatty acid composition in mammary adipose tissue using deep neural network with unsupervised training (under review)
