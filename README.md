# FAC-Net
Fatty Acid Composition - Deep Neural Network

facnet.py: the python script for FAC-Net described in Chaudhary et al (MRM 2024)
example_data.h5: multi-gradient echo images used for Figure 3 of Chaudhary et al (MRM 2024)

To run the script, download both facnet.py and example_data.h5 to a folder. 
Use the following command to run the facnet with the example data.

python facnet.py --fac_h5 "example_data" --n_epochs 7000 --user_learning_rate 0.0008

## Reference
Chaudhary S, Lane EG, Levy A, McGrath A, Mema E, Reichmann M, Dodelzon K, Simon K, Chang E, Nickel MD, Moy L, Drotman M, Kim SG, Estimation of fatty acid composition in mammary adipose tissue using deep neural network with unsupervised training (Magn Reson Med. 2024 Dec 6. doi: 10.1002/mrm.30401. Online ahead of print.)
