#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import math
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__

import random as rn
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
  Input, BatchNormalization, Activation, Dense, Dropout, Flatten, 
  Lambda, UpSampling2D, RepeatVector, Reshape, Conv2D, Conv2DTranspose,
  MaxPooling2D, GlobalMaxPool2D, concatenate
)
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import (
  EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam


np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)


def network(input_train_tm_tens,input_train_tp_tens,dfat_train_t1_tens,dfat_train_t2_tens,dfat_train_t3_tens,dfat_train_t4_tens,dfat_train_t5_tens,dfat_train_t6_tens, dfat_train_t7_tens,dfat_train_t8_tens,dfat_train_t9_tens,dfat_train_t10_tens,te_train_t_tens,p_tens):

   n_filters = 32
   kernel_size =2


   c1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', kernel_initializer = initializers.he_uniform(seed=None),padding = 'same')(input_train_tm_tens)
   c1 = BatchNormalization()(c1)
   p1 = MaxPooling2D((2, 2)) (c1)

   c2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (p1)
   c2 = BatchNormalization()(c2)
   p2 = MaxPooling2D((2, 2))(c2)

   c3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (p2)
   c3 = BatchNormalization()(c3)
   p3 = MaxPooling2D((2, 2)) (c3)

   c4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (p3)
   c4 = BatchNormalization()(c4)
   p4 = MaxPooling2D((2, 2)) (c4)

   c5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (p4)
   c5 = BatchNormalization()(c5)
   p5 = MaxPooling2D((2, 2)) (c5)

   c6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (p5)
   c6 = BatchNormalization()(c6)
   c6 = UpSampling2D(size=(2, 2), data_format=None) (c6)

   u7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c6)
   u7 = concatenate([u7, c5])
   c7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (u7)
   c7 = BatchNormalization()(c7)
   c7 = UpSampling2D(size=(2, 2), data_format=None) (c7)

   u8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c7)
   u8 = concatenate([u8, c4],axis=3)
   c8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (u8)
   c8 = BatchNormalization()(c8)
   c8 = UpSampling2D(size=(2, 2), data_format=None) (c8)

   u9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c8)
   u9 = concatenate([u9, c3],axis=3)
   c9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (u9)
   c9 = BatchNormalization()(c9)
   c9 = UpSampling2D(size=(2, 2), data_format=None) (c9)

   u10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c9)
   u10 = concatenate([u10, c2],axis=3)
   c10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (u10)
   c10 = BatchNormalization()(c10)
   c10 = UpSampling2D(size=(2, 2), data_format=None) (c10)

   u11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c10)
   u11 = concatenate([u11, c1],axis=3)
   c11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (u11)
   c11 = BatchNormalization()(c11)
   #c11 = Dropout(0.1)(c11)
   #c11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='HeNormal', padding='same')(c11)
   c11 = UpSampling2D(size=(1, 1), data_format=None) (c11)

   output_pred1 = Conv2D(5,(1, 1), activation='linear') (c11)


   cc1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = initializers.he_uniform(seed=0))(input_train_tp_tens)
   cc1 = BatchNormalization()(cc1)
   pp1 = MaxPooling2D((2, 2)) (cc1)

   cc2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (pp1)
   cc2 = BatchNormalization()(cc2)
   pp2 = MaxPooling2D((2, 2))(cc2)

   cc3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (pp2)
   cc3 = BatchNormalization()(cc3)
   pp3 = MaxPooling2D((2, 2)) (cc3)

   cc4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (pp3)
   cc4 = BatchNormalization()(cc4)
   pp4 = MaxPooling2D((2, 2)) (cc4)

   cc5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (pp4)
   cc5 = BatchNormalization()(cc5)
   pp5 = MaxPooling2D((2, 2)) (cc5)

   cc6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (pp5)
   cc6 = BatchNormalization()(cc6)
   cc6 = UpSampling2D(size=(2, 2), data_format=None) (cc6)

   uu7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc6)
   uu7 = concatenate([uu7, cc5])
   cc7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (uu7)
   cc7 = BatchNormalization()(cc7)
   cc7 = UpSampling2D(size=(2, 2), data_format=None) (cc7)

   uu8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc7)
   uu8 = concatenate([uu8, cc4])
   cc8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (uu8)
   cc8 = BatchNormalization()(cc8)
   cc8 = UpSampling2D(size=(2, 2), data_format=None) (cc8)


   uu9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc8)
   uu9 = concatenate([uu9, cc3])
   cc9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (uu9)
   cc9 = BatchNormalization()(cc9)
   cc9 = UpSampling2D(size=(2, 2), data_format=None) (cc9)

   uu10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc9)
   uu10 = concatenate([uu10, cc2])
   cc10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None),padding='same') (uu10)
   cc10 = BatchNormalization()(cc10)
   cc10 = UpSampling2D(size=(2, 2), data_format=None) (cc10)

   uu11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc10)
   uu11 = concatenate([uu11, cc1], axis=3)
   cc11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=None), padding='same') (uu11)
   cc11 = BatchNormalization()(cc11)
   cc11 = UpSampling2D(size=(1, 1), data_format=None) (cc11)

   output_pred2 = Conv2D(4,(1, 1), activation='linear') (cc11)

   output_pred = concatenate([output_pred1,output_pred2])

   model = Model(inputs=[input_train_tm_tens,input_train_tp_tens,dfat_train_t1_tens,dfat_train_t2_tens,dfat_train_t3_tens,dfat_train_t4_tens,dfat_train_t5_tens,dfat_train_t6_tens, dfat_train_t7_tens,dfat_train_t8_tens,dfat_train_t9_tens,dfat_train_t10_tens,te_train_t_tens, p_tens], outputs=[output_pred])


   return model,output_pred
   
   
def loss1(input_train_tm_tens,input_train_tp_tens, dfat_train_t1_tens,dfat_train_t2_tens,dfat_train_t3_tens,dfat_train_t4_tens,dfat_train_t5_tens,dfat_train_t6_tens,dfat_train_t7_tens,dfat_train_t8_tens,dfat_train_t9_tens,dfat_train_t10_tens,te_train_t_tens,p_tens,w_mean_r_6,w_std_r_6,f_mean_r_6,f_std_r_6,w_mean_i_6,w_std_i_6,f_mean_i_6,f_std_i_6,frq_mean_6,frq_std_6,r2_mean_6,r2_std_6,output_pred):

    pi = tf.constant(math.pi)
    k1=tf.cast(tf.constant(4.0), tf.float32)


    wat_r = output_pred[:,:,:,0]
    watt_r_0 = tf.expand_dims(wat_r,3)
    watt_r_1= tf.repeat(watt_r_0,16,3)
    watt_r_2 = tf.math.add(tf.math.scalar_mul(w_std_r_6,watt_r_1),w_mean_r_6)
    watt_r=   tf.math.abs(watt_r_2)

    fat_r = output_pred[:,:,:,1]
    fatt_r_0 = tf.expand_dims(fat_r,3)
    fatt_r_1= tf.repeat(fatt_r_0,16,3)
    fatt_r_2 = tf.math.add(tf.math.scalar_mul(f_std_r_6,fatt_r_1),f_mean_r_6)
    fatt_r=tf.math.abs(fatt_r_2)

    r2 = output_pred[:,:,:,2]
    r2t = tf.expand_dims(r2,3)
    r2t= tf.repeat(r2t,16,3)
    r2t = tf.math.add(tf.math.scalar_mul(r2_std_6,r2t),r2_mean_6)

    ndb_r= output_pred[:,:,:,3]
    ndbb_r=tf.expand_dims(ndb_r,3)
    ndbb_r=tf.repeat(ndbb_r,16,3)
    ndbb_rr_0=tf.math.add(tf.math.scalar_mul(ndb_std_r_6,ndbb_r),ndb_mean_r_6)
    ndbb_rr=tf.math.abs(ndbb_rr_0)

    nmidb_r= output_pred[:,:,:,4]
    nmidbb_r=tf.expand_dims(nmidb_r,3)
    nmidbb_r=tf.repeat(nmidbb_r,16,3)
    nmidbb_rr_0=tf.math.add(tf.math.scalar_mul(nmidb_std_r_6,nmidbb_r),nmidb_mean_r_6)
    nmidbb_rr=tf.math.abs(nmidbb_rr_0)                     


    wat_i = output_pred[:,:,:,5]
    watt_i = tf.expand_dims(wat_i,3)
    watt_i= tf.repeat(watt_i,16,3)
    watt_i = tf.math.add(tf.math.scalar_mul(w_std_i_6,watt_i),w_mean_i_6)

    fat_i = output_pred[:,:,:,6]
    fatt_i = tf.expand_dims(fat_i,3)
    fatt_i= tf.repeat(fatt_i,16,3)
    fatt_i = tf.math.add(tf.math.scalar_mul(f_std_i_6,fatt_i),f_mean_i_6)

    frq = output_pred[:,:,:,7]
    frqt = tf.expand_dims(frq,3)
    frqt= tf.repeat(frqt,16,3)
    frqt = tf.math.add(tf.math.scalar_mul(frq_std_6,frqt),frq_mean_6)


    p=output_pred[:,:,:,8]
    p_0=tf.shape(p, tf.int32)
    p_3=tf.zeros(p_0,tf.float32)
    p_1=tf.stack([p_3 ,p,p_3 ,p, p_3,p,p_3 , p,p_3 , p,p_3 , p, p_3, p,p_3 , p],axis=3)
    p_2=tf.math.add(tf.math.scalar_mul(p_std_6,p_1),p_mean_6)

    frq_tv_0=tf.image.total_variation(frq)

    watt_c =  tf.cast(tf.complex(watt_r,0*watt_r), tf.complex64)
    fatt_c=  tf.cast(tf.complex(fatt_r,0*fatt_r), tf.complex64)
    ndbb_c =  tf.cast(tf.complex(ndbb_rr,0*ndbb_rr), tf.complex64)
    nmidbb_c =  tf.cast(tf.complex(nmidbb_rr,0*nmidbb_rr), tf.complex64)


    watt_ci =  tf.cast(tf.complex(watt_i,0*watt_i), tf.complex64)
    fatt_ci =  tf.cast(tf.complex(fatt_i,0*fatt_i), tf.complex64)

    r2t_c =  tf.cast(tf.complex(r2t,0*r2t), tf.complex64)
    frqt_c = tf.cast(tf.complex(frqt,0*frqt),tf.complex64)

    dfat_train_t1_c = tf.cast(tf.complex(dfat_train_t1_tens,0*dfat_train_t1_tens),tf.complex64)
    dfat_train_t2_c = tf.cast(tf.complex(dfat_train_t2_tens,0*dfat_train_t2_tens),tf.complex64)
    dfat_train_t3_c = tf.cast(tf.complex(dfat_train_t3_tens,0*dfat_train_t3_tens),tf.complex64)
    dfat_train_t4_c = tf.cast(tf.complex(dfat_train_t4_tens,0*dfat_train_t4_tens),tf.complex64)
    dfat_train_t5_c = tf.cast(tf.complex(dfat_train_t5_tens,0*dfat_train_t5_tens),tf.complex64)
    dfat_train_t6_c = tf.cast(tf.complex(dfat_train_t6_tens,0*dfat_train_t6_tens),tf.complex64)
    dfat_train_t7_c = tf.cast(tf.complex(dfat_train_t7_tens,0*dfat_train_t7_tens),tf.complex64)
    dfat_train_t8_c = tf.cast(tf.complex(dfat_train_t8_tens,0*dfat_train_t8_tens),tf.complex64)
    dfat_train_t9_c = tf.cast(tf.complex(dfat_train_t9_tens,0*dfat_train_t9_tens),tf.complex64)
    dfat_train_t10_c = tf.cast(tf.complex(dfat_train_t10_tens,0*dfat_train_t10_tens),tf.complex64)


    te_train_t_c = tf.cast(tf.complex(te_train_t_tens,0*te_train_t_tens),tf.complex64)
    p_c=tf.cast(tf.complex(p_2,0*p_2),tf.complex64)

    pi_cmp =  tf.cast(tf.complex(pi,0*pi), tf.complex64)

    n1=tf.cast((9*(tf.exp(-1j*2*pi_cmp*dfat_train_t1_c*te_train_t_c))),tf.complex64)
    n2=tf.cast((76.8-6.5*(ndbb_c)+2*(nmidbb_c))*(tf.exp(-1j*2*pi_cmp*dfat_train_t2_c*te_train_t_c)),tf.complex64)
    n3=tf.cast((6*(tf.exp(-1j*2*pi_cmp*dfat_train_t3_c*te_train_t_c))),tf.complex64)
    n4=tf.cast(((((ndbb_c)-(nmidbb_c))*4)*(tf.exp(-1j*2*pi_cmp*dfat_train_t4_c*te_train_t_c))),tf.complex64)
    n5=tf.cast((6*(tf.exp(-1j*2*pi_cmp*dfat_train_t5_c*te_train_t_c))),tf.complex64)
    n6=tf.cast((((nmidbb_c)*2)*(tf.exp(-1j*2*pi_cmp*dfat_train_t6_c*te_train_t_c))),tf.complex64)
    n7=tf.cast((2*(tf.exp(-1j*2*pi_cmp*dfat_train_t7_c*te_train_t_c))),tf.complex64)
    n8=tf.cast((2*(tf.exp(-1j*2*pi_cmp*dfat_train_t8_c*te_train_t_c))),tf.complex64)
    n9=tf.cast((1*(tf.exp(-1j*2*pi_cmp*dfat_train_t9_c*te_train_t_c))),tf.complex64)
    n10=tf.cast((((ndbb_c)*2)*(tf.exp(-1j*2*pi_cmp*dfat_train_t10_c*te_train_t_c))),tf.complex64)


    m00=tf.math.abs(n1)+tf.math.abs(n2)+tf.math.abs(n3)+tf.math.abs(n4)+tf.math.abs(n5)+tf.math.abs(n6)+tf.math.abs(n7)+tf.math.abs(n8)+tf.math.abs(n9)+tf.math.abs(n10)
    m=tf.cast(1/m00,tf.complex64)


    signal = tf.cast(((watt_c)*tf.exp(1j*watt_ci) + (tf.exp(1j*fatt_ci)*(fatt_c))*(m*(n1+n2+n3+n4+n5+n6+n7+n8+n9+n10)))*tf.exp(-1*r2t_c*te_train_t_c)*tf.exp(1j*p_c)*tf.exp(-1j*2*pi_cmp*frqt_c*te_train_t_c),tf.complex64)
    input_train_t_mag = input_train_tm_tens
    input_train_t_phs = input_train_tp_tens

    gt_input_train2 = tf.cast(tf.multiply(tf.complex(input_train_t_mag,0*input_train_t_mag),tf.exp(1j*tf.complex(input_train_t_phs,0*input_train_t_phs))),tf.complex64)

    loss= tf.linalg.norm(tf.abs(gt_input_train2-signal)) + 0.0001*frq_tv_0

    return loss
                   

if __name__ == "__main__":

    # %% parse
    parser = argparse.ArgumentParser(description='run FAC-Net.')

    parser.add_argument('--dir_in', type=str, default='./',
                        help='the directory where h5 dicom files are stored')

    parser.add_argument('--dir_out', type=str, default='./',
                        help='the directory where h5 file will be stored')

    parser.add_argument('--fac_h5', type=str, default='data',
                        help='the h5 file name to save the data')

    parser.add_argument('--n_epochs', type=int, default=500,
                        help='the number of epochs')

    parser.add_argument('--user_learning_rate', type=float, default=0.0008,
                        help='learning rate')

    args = parser.parse_args()

    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)

    print('> dir_in = ', args.dir_in)
    print('> dir_out = ', args.dir_out)
    print('> fac_h5 = ', args.fac_h5)
    print('> n_epochs = ', args.n_epochs)
    print('> learning_rate = ', args.user_learning_rate)
       
    N_epochs = args.n_epochs
    user_learning_rate = args.user_learning_rate

    OUT_DIR = args.dir_out
    try:
        os.mkdir(OUT_DIR)
    except OSError as error:
        print(error)
        
    f = h5py.File(args.dir_in + '/' + args.fac_h5 + '.h5', 'r')
    mag_1 = f['mag'][:]
    ph_1 = f['pha'][:]
    TE    = f['/pha'].attrs['te']
    c_frq = f['/pha'].attrs['cfreq']    
    f.close()     
      
    
    print('>> Mag : ', mag_1.shape)
    print('>> Pha : ', ph_1.shape)
    print('>> TE : ', TE)
    print('>> c_frq : ', c_frq)

    NE = len(TE)

    outresults_15=[]
    water_15=[]
    fat_15=[]
    ndbb_15=[]
    nmidbb_15=[]
    cl_15=[]
    frq_15=[]
    R2_15=[]
    p2_15=[]

        
    input_m=mag_1.copy()
    input_ph=ph_1.copy()

    input_train_m=input_m.copy()
    input_train_ph=input_ph.copy()

    m_0=input_train_m [:,:,0].copy()
    me_m0=np.mean(m_0)

    for i8,j8 in enumerate(m_0):
        for x,y in enumerate(j8):
            if y<=me_m0:
                j8[x]=0.0
            else:
                j8[x]=1.0

    mask_0=m_0.copy()
    mask_6=mask_0.copy()
    mask_12=[]
    for x,y in enumerate(TE):
        mask_8=mask_6.copy()
        mask_8[:,:]=y
        k=mask_8
        mask_12.append(k)
        mask_15=np.stack(mask_12,axis=2)

    breast_image_te_1=mask_15.copy()
    breast_image_te=breast_image_te_1.copy()

    b_ff_1=mask_0.copy()
    b_fff_1=1.0-b_ff_1

    breast_image_fat_0=b_fff_1.copy()

    A_peak=4.7-0.90
    B_peak=4.7-1.30
    C_peak=4.7-1.59
    D_peak=4.7-2.03
    E_peak=4.7-2.25
    F_peak=4.7-2.77
    G_peak=4.7-4.1
    H_peak=4.7-4.3
    I_peak=4.7-5.21
    J_peak=4.7-5.31

    breast_image_fat_01=b_fff_1.copy()

    c_frq_1=[A_peak,B_peak,C_peak,D_peak,E_peak,F_peak,G_peak,H_peak,I_peak,J_peak]

    c_frq_2=[]

    for i9,j9 in enumerate(c_frq_1):
        x=(c_frq)*(j9/1000000)
        c_frq_2.append(x)


    breast_image_fat_011=breast_image_fat_01.copy()
    breast_image_fat_014=[]
    for i10,j10 in enumerate(c_frq_2):
        breast_image_fat_012=breast_image_fat_011.copy()
        breast_image_fat_012[:,:]=j10
        k=breast_image_fat_012
        breast_image_fat_014.append(k)
    breast_image_fat_015=[]
    for i11,j11 in enumerate(breast_image_fat_014):
        j1_11=np.repeat(j11[None,...],NE,axis=0)
        j2_22=j1_11.transpose((1,2,0))
        breast_image_fat_015.append(j2_22)


    p_mask_0=mask_0.copy()
    for i12, j12 in enumerate(p_mask_0):
        for x, y in enumerate(j12):
            j12[x]= 0.0

    p_mask_1=np.repeat(p_mask_0[None,...],NE,axis=0)
    p_mask_00=p_mask_1.transpose((1,2,0))


    input_train_m=input_m.copy()
    input_train_ph=input_ph.copy()
    input_train_mm=input_m.copy()
    input_train_phph=input_ph.copy()

    te_train_t=breast_image_te.copy()
    input_train_tm_tm=input_train_m.copy()
    input_train_tp_tp=input_train_ph.copy()
    input_test_tm_tm=input_train_m.copy()
    input_test_tp_tp=input_train_ph.copy()

    w_mean_r_6 = float(0.2)
    w_std_r_6= float(0.1)
    f_mean_r_6= float(0.1)
    f_std_r_6= float(0.2)
    frq_mean_6= float(30)
    frq_std_6= float(30)
    r2_mean_6= float(80)
    r2_std_6= float(50)
    w_mean_i_6= float(-1)
    w_std_i_6= float(1)
    f_mean_i_6=float(-1)
    f_std_i_6= float(1)
    p_std_6=float(0.2)
    p_mean_6=float(0.1)
    ndb_std_r_6=float(0.2)
    ndb_mean_r_6=float(0.1)
    nmidb_std_r_6=float(0.2)
    nmidb_mean_r_6=float(0.1)
    cl_std_r_6=float(0.1)
    cl_mean_r_6=float(17.0)

    te_train_t=breast_image_te.copy()
    outresults_11=[]
    history_11=[]

    dfat_train_t1=breast_image_fat_015[0]
    dfat_train_t2=breast_image_fat_015[1]
    dfat_train_t3=breast_image_fat_015[2]
    dfat_train_t4=breast_image_fat_015[3]
    dfat_train_t5=breast_image_fat_015[4]
    dfat_train_t6=breast_image_fat_015[5]
    dfat_train_t7=breast_image_fat_015[6]
    dfat_train_t8=breast_image_fat_015[7]
    dfat_train_t9=breast_image_fat_015[8]
    dfat_train_t10=breast_image_fat_015[9]

    input_shape = (input_train_tm_tm[:,:, :]).shape
    input_train_tm_tens = Input(input_shape, name='input_train_tm_tm')
    input_train_tp_tens = Input(input_shape, name='input_train_tp_tp')
    input_shape = (dfat_train_t1[:,:, :]).shape
    dfat_train_t1_tens = Input(input_shape, name='dfat_train_t1')
    input_shape = (dfat_train_t2[:,:, :]).shape
    dfat_train_t2_tens = Input(input_shape, name='dfat_train_t2')
    input_shape = (dfat_train_t3[:,:, :]).shape
    dfat_train_t3_tens = Input(input_shape, name='dfat_train_t3')
    input_shape = (dfat_train_t4[:,:, :]).shape
    dfat_train_t4_tens = Input(input_shape, name='dfat_train_t4')
    input_shape = (dfat_train_t5[:,:, :]).shape
    dfat_train_t5_tens = Input(input_shape, name='dfat_train_t5')
    input_shape = (dfat_train_t6[:,:, :]).shape
    dfat_train_t6_tens = Input(input_shape, name='dfat_train_t6')
    input_shape = (dfat_train_t7[:,:, :]).shape
    dfat_train_t7_tens = Input(input_shape, name='dfat_train_t7')
    input_shape = (dfat_train_t8[:,:, :]).shape
    dfat_train_t8_tens = Input(input_shape, name='dfat_train_t8')
    input_shape = (dfat_train_t9[:,:, :]).shape
    dfat_train_t9_tens = Input(input_shape, name='dfat_train_t9')
    input_shape = (dfat_train_t10[:,:, :]).shape
    dfat_train_t10_tens = Input(input_shape, name='dfat_train_t10')
    input_shape = (te_train_t[:,:, :]).shape
    te_train_t_tens = Input(input_shape, name='te_train_t')
    p_tens=Input(input_shape, name='p_mask_00')
    print(input_train_tm_tens.shape)


    model,output_pred = network(input_train_tm_tens,input_train_tp_tens, dfat_train_t1_tens,dfat_train_t2_tens,dfat_train_t3_tens,dfat_train_t4_tens,dfat_train_t5_tens,dfat_train_t6_tens,dfat_train_t7_tens,dfat_train_t8_tens,dfat_train_t9_tens,dfat_train_t10_tens,te_train_t_tens, p_tens)

    custom_loss3 = model.add_loss(1*loss1(input_train_tm_tens,input_train_tp_tens, dfat_train_t1_tens, dfat_train_t2_tens, dfat_train_t3_tens, dfat_train_t4_tens, dfat_train_t5_tens, dfat_train_t6_tens,dfat_train_t7_tens,dfat_train_t8_tens,dfat_train_t9_tens,dfat_train_t10_tens,te_train_t_tens,p_tens,w_mean_r_6,w_std_r_6,f_mean_r_6,f_std_r_6,w_mean_i_6,w_std_i_6,f_mean_i_6,f_std_i_6,frq_mean_6,frq_std_6,r2_mean_6,r2_std_6,output_pred))

    input_test_m_1=input_test_tm_tm.copy()
    input_test_m_11=np.expand_dims(input_test_m_1,axis=0)
    input_test_ph_1=input_test_tp_tp.copy()
    input_test_ph_11=np.expand_dims(input_test_ph_1,axis=0)

    input_train_tm_1=input_train_tm_tm.copy()
    input_train_tm_11=np.expand_dims(input_train_tm_1,axis=0)

    input_train_tp_1=input_train_tp_tp.copy()
    input_train_tp_11=np.expand_dims(input_train_tp_1,axis=0)

    te_train_t_1=te_train_t.copy()
    te_train_t_11=np.expand_dims(te_train_t_1,axis=0)

    p_mask_1=p_mask_00.copy()
    p_mask_11=np.expand_dims(p_mask_1,axis=0)

    dfat_train_t1_1=dfat_train_t1.copy()
    dfat_train_t1_11=np.expand_dims(dfat_train_t1_1,axis=0)

    dfat_train_t2_1=dfat_train_t2.copy()
    dfat_train_t2_11=np.expand_dims(dfat_train_t2_1,axis=0)

    dfat_train_t3_1=dfat_train_t3.copy()
    dfat_train_t3_11=np.expand_dims(dfat_train_t3_1,axis=0)

    dfat_train_t4_1=dfat_train_t4.copy()
    dfat_train_t4_11=np.expand_dims(dfat_train_t4_1,axis=0)

    dfat_train_t5_1=dfat_train_t5.copy()
    dfat_train_t5_11=np.expand_dims(dfat_train_t5_1,axis=0)

    dfat_train_t6_1=dfat_train_t6.copy()
    dfat_train_t6_11=np.expand_dims(dfat_train_t6_1,axis=0)

    dfat_train_t7_1=dfat_train_t7.copy()
    dfat_train_t7_11=np.expand_dims(dfat_train_t7_1,axis=0)

    dfat_train_t8_1=dfat_train_t8.copy()
    dfat_train_t8_11=np.expand_dims(dfat_train_t8_1,axis=0)

    dfat_train_t9_1=dfat_train_t9.copy()
    dfat_train_t9_11=np.expand_dims(dfat_train_t9_1,axis=0)

    dfat_train_t10_1=dfat_train_t10.copy()
    dfat_train_t10_11=np.expand_dims(dfat_train_t10_1,axis=0)


    model.compile(optimizer=Adam(learning_rate=user_learning_rate), loss=custom_loss3)
    filepath = "saved_weights.h5"

    callbacks = [ReduceLROnPlateau(factor=0.1, patience=3, learning_rate=0.0001, verbose=1), ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True, mode='auto',save_freq=1)]

    history = model.fit([input_train_tm_11,input_train_tp_11,dfat_train_t1_11,dfat_train_t2_11,dfat_train_t3_11,dfat_train_t4_11,dfat_train_t5_11,dfat_train_t6_11,dfat_train_t7_11,dfat_train_t8_11,dfat_train_t9_11,dfat_train_t10_11,te_train_t_11,p_mask_11], callbacks=callbacks, validation_split=0, batch_size=2, epochs=N_epochs,shuffle=False)

    outresults = model.predict([input_test_m_11,input_test_ph_11,dfat_train_t1_11,dfat_train_t2_11,dfat_train_t3_11,dfat_train_t4_11,dfat_train_t5_11,dfat_train_t6_11,dfat_train_t7_11,dfat_train_t8_11,dfat_train_t9_11,dfat_train_t10_11, te_train_t_11,p_mask_11],batch_size=2)


    outresults_1=outresults.copy()
    outresults_2=outresults_1.transpose((1,2,0,3))
    print(outresults_2.shape)
    w_mean_r_6 = float(0.2)
    w_std_r_6= float(0.1)
    f_mean_r_6= float(0.1)
    f_std_r_6= float(0.2)
    frq_mean_6= float(30)
    frq_std_6= float(30)
    r2_mean_6= float(80)
    r2_std_6= float(50)
    w_mean_i_6= float(-1)
    w_std_i_6= float(1)
    f_mean_i_6=float(-1)
    f_std_i_6= float(1)
    p_std_6=float(0.2)
    p_mean_6=float(0.1)
    ndb_std_r_6=float(0.2)
    ndb_mean_r_6=float(0.1)
    nmidb_std_r_6=float(0.2)
    nmidb_mean_r_6=float(0.1)


    water_abs_11_0=(outresults_2[:,:,:,0]*w_std_r_6)+w_mean_r_6
    water_abs_11=np.absolute(water_abs_11_0)
    print(water_abs_11.shape)

    fat_abs_pd_11_0=(outresults_2[:,:,:,1]*f_std_r_6)+f_mean_r_6
    fat_abs_pd_11=np.absolute(fat_abs_pd_11_0)
    print(fat_abs_pd_11.shape)

    R2_11=(outresults_2[:,:,:,2]*r2_std_6)+r2_mean_6

    ndbb_11_0=(outresults_2[:,:,:,3]*ndb_std_r_6)+ndb_mean_r_6
    ndbb_11=np.absolute(ndbb_11_0)

    nmidbb_11_0=(outresults_2[:,:,:,4]*nmidb_std_r_6)+nmidb_mean_r_6
    nmidbb_11=np.absolute(nmidbb_11_0)

    frq_11=(outresults_2[:,:,:,7]*frq_std_6)+frq_mean_6

    p2_11=(outresults_2[:,:,:,8]*f_std_i_6)+f_mean_i_6

    s_0=input_train_m [:,:,1].copy()
    for x1,x2 in enumerate(s_0):
        for y1,y2 in enumerate(x2):
            ss_0=np.mean(s_0)
            if y2<=ss_0:
                x2[y1]=0.0
            else:
                x2[y1]=1.0

    s_00=np.expand_dims(s_0,2)
    s_01=np.repeat(s_00,1,2)

    water_abs_18=water_abs_11*s_01
    fat_abs_12=fat_abs_pd_11*s_01
    ndbb_12=ndbb_11*s_01
    nmidbb_12=nmidbb_11*s_01
    R2_12=R2_11*s_01
    p2_12=p2_11*s_01
    frq_12=frq_11*s_01


    water_15.append(water_abs_18)
    fat_15.append(fat_abs_12)
    ndbb_15.append(ndbb_12)
    nmidbb_15.append(nmidbb_12)
    R2_15.append(R2_12)
    p2_15.append(p2_12)
    frq_15.append(frq_12)
    outresults_15.append(outresults_2)


    # save result
    f = h5py.File(OUT_DIR + '/' + args.fac_h5 + '_fac.h5', 'w')
    dset = f.create_dataset('water', data=water_15)
    dset = f.create_dataset('fat', data=fat_15)
    dset = f.create_dataset('ndb', data=ndbb_15)
    dset = f.create_dataset('nmidb', data=nmidbb_15)
    dset = f.create_dataset('frq', data=frq_15)
    dset = f.create_dataset('r2s', data=R2_15)
    dset = f.create_dataset('p2', data=p2_15)
    dset = f.create_dataset('outresults', data=outresults_15)
    f.close()
        
    print('> done')
    




