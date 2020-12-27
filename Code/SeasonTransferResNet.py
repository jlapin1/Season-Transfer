# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:15:35 2020

@author: joell
"""

import numpy as np
import tensorflow as tf
import os
import PIL
import sys
import time
import matplotlib.pyplot as plt

# tf.pad(tensor, [[0,0],[hpad,hpad],[wpad,wpad],[0,0]],'REFLECT')

dims = (256, 256, 3)

BATCH_SIZE = 1
LAMBDA = 10

initializer = tf.random_normal_initializer(0., 0.02)

def normalize(im):
    im = (im-127.5)/127.5
    return im

def random_jitter(im):
    im = im.resize((286,286))
    
    rnd = np.random.randint(30, size=(2,))
    
    im = im.crop([rnd[0], rnd[1], rnd[0]+256, rnd[1]+256])
    
    if np.random.rand() < 0.5:
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    
    return im

# Load datasets
autumn_path = '/mnt/c/Users/joell/MyDatasets/Seasons/Cropped/Autumn/'
autumn_files = os.listdir(autumn_path)
summer_path = '/mnt/c/Users/joell/MyDatasets/Seasons/Cropped/Summer/'
summer_files = os.listdir(summer_path)
tot_files = np.min([len(autumn_files), len(summer_files)])

AD = np.zeros((tot_files,)+dims, dtype=np.float32)
SD = np.zeros((tot_files,)+dims, dtype=np.float32)
for cnt,(afile,sfile) in enumerate(zip(autumn_files,summer_files)):
    aim = PIL.Image.open(autumn_path+afile)
    sim = PIL.Image.open(summer_path+sfile)
    
    aim = random_jitter(aim)
    sim = random_jitter(sim)
    
    AD[cnt] = normalize(np.array(aim))
    SD[cnt] = normalize(np.array(sim))
perm = np.random.permutation(tot_files)
AD = AD[perm]
SD = SD[perm]
AUT = tf.constant(AD, tf.float32) # x
SUM = tf.constant(SD, tf.float32) # y

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, in_shape):
        super(InstanceNormalization, self).__init__()
        self.D = in_shape
        self.eps = tf.Variable(tf.zeros(self.D, dtype=tf.float32), trainable=True)
        self.gamma = tf.Variable(tf.ones(self.D, dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros(self.D, dtype=tf.float32), trainable=True)
    
    def call(self, inp):
        mean = tf.reduce_mean(inp, axis=[1,2])
        mean = mean[:, tf.newaxis, tf.newaxis, :]
        
        im = inp-mean
        
        sig2 = tf.reduce_mean(im**2, axis=[1,2])
        sig2 = sig2[:, tf.newaxis, tf.newaxis, :]
        out = im/tf.sqrt(sig2+self.eps[:, tf.newaxis, tf.newaxis, :])
        out = tf.multiply(self.gamma[:, tf.newaxis, tf.newaxis, :], out) + self.beta[:, tf.newaxis, tf.newaxis, :]
        return out

def downsample(filters, size, in_shape=None, apply_instnorm=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    if apply_instnorm:
        # model.add(InstanceNormalization(in_shape))
        model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.ReLU())
    
    return model

def upsample(filters, size, in_shape, apply_dropout=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    
    # model.add(InstanceNormalization(in_shape))
    model.add(tf.keras.layers.BatchNormalization())
    
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.ReLU())
    
    return model

def c7s1(k, apply_norm=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=k, kernel_size=7, strides=1, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model

def dk(k):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=k, kernel_size=3, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model

class Rk(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Rk, self).__init__()
        #self.filters=filters
        self.block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, 
                                   kernel_initializer=initializer, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1,
                                   kernel_initializer=initializer, padding='same'),
            tf.keras.layers.BatchNormalization()
            ])
    def call(self, inp):
        return self.block(inp)+inp

def uk(k, dropout=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(filters=k, kernel_size=3, strides=2,
                                              kernel_initializer=initializer, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    if dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.ReLU())
    return model

def Generator(res_blocks):
    inim = tf.keras.layers.Input(shape=dims) # bs, 256, 256, 3
    
    downstack = [
        c7s1(64, apply_norm=False),
        dk(128),
        dk(256),
        ]
    resstack = []
    for _ in range(res_blocks):
        resstack.append(Rk(256))
    upstack = [
        uk(128),
        uk(64)
        ]
    
    last = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, kernel_initializer=initializer,
                                  padding='same', activation='tanh')
    # operations
    out = inim
    
    for layer in downstack:
        out = layer(out)
    for layer in resstack:
        out = layer(out)
    for layer in upstack:
        out = layer(out)
    
    out = last(out)
    
    return tf.keras.Model(inputs=inim, outputs=out)

def Discriminator():
    inp = tf.keras.layers.Input(shape=dims) # bs, 256, 256, 3
    
    out = downsample(64, 4, (BATCH_SIZE, 64), False)(inp) # bs, 128, 128, 64
    out = downsample(128, 4, (BATCH_SIZE, 128))(out) # bs, 64, 64, 128
    out = downsample(256, 4, (BATCH_SIZE, 256))(out) # bs, 32, 32, 256
    
    out = tf.keras.layers.ZeroPadding2D((1,1))(out) # bs, 34, 34, 256
    
    out = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding='valid',
                           kernel_initializer=initializer)(out) # bs, 31, 31, 512
    
    # out = InstanceNormalization((BATCH_SIZE, 512))(out)
    out = tf.keras.layers.BatchNormalization()(out)
    
    out = tf.keras.layers.LeakyReLU(0.2)(out)
    
    out = tf.keras.layers.ZeroPadding2D()(out) # bs, 33, 33, 512
    
    out = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='valid',
                                 kernel_initializer=initializer)(out) # bs, 30, 30, 1
    
    return tf.keras.Model(inputs=inp, outputs=out)

#pip install -q git+https://github.com/tensorflow/examples.git
#from tensorflow_examples.models.pix2pix import pix2pix

genxy = Generator(6)#pix2pix.unet_generator(3, norm_type='instancenorm')
genyx = Generator(6)#pix2pix.unet_generator(3, norm_type='instancenorm')
disx = Discriminator()#pix2pix.discriminator(norm_type='instancenorm', target=False)
disy = Discriminator()#pix2pix.discriminator(norm_type='instancenorm', target=False)

def generator_loss(gens):
    gen_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(gens), gens, from_logits=True))
    return gen_loss

def discriminator_loss(reals, gens):
    real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(reals), reals, from_logits=True))
    gen_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(gens), gens, from_logits=True))
    tot_loss = real_loss + gen_loss
    return tot_loss, real_loss, gen_loss

def cycle_loss(GEN, orig):
    l1_loss = tf.reduce_mean(tf.abs(GEN-orig), axis=[1,2,3])
    return LAMBDA*l1_loss

def train(train_tuple, epochs=10, bs=1, lr=2e-4, svwts=False):
    
    print("Starting training session lasting %d epochs"%(epochs))
    
    trX,trY = train_tuple
    
    trtot = trX.shape[0]
    
    steps = trtot//bs + 1 if trtot%bs!=0 else trtot//bs
    
    gxyOpt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    gyxOpt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    dxOpt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    dyOpt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    for m in range(epochs):
        tots = np.zeros((7))
        start_epoch = time.time()
        for n in range(steps):
            start_step = time.time()
            
            first = n*bs
            last = (n+1)*bs
            amt = last-first
            
            inpx = trX[first:last]
            inpy = trY[first:last]
            
            with tf.GradientTape(persistent=True) as tape:
                Ys= genxy(inpx, training=True)
                Xcyc = genyx(Ys, training=True)
                Xs = genyx(inpy, training=True)
                Ycyc = genxy(Xs, training=True)
                Ysame = genxy(inpy, training=True)
                Xsame = genyx(inpx, training=True)
                
                Yreal = disy(inpy, training=True)
                X2Y = disy(Ys, training=True)
                Xreal = disx(inpx, training=True)
                Y2X = disx(Xs, training=True)
                
                genxy_loss = generator_loss(X2Y)
                genyx_loss = generator_loss(Y2X)
                
                ycyc_loss = cycle_loss(Ycyc, inpy)
                xcyc_loss = cycle_loss(Xcyc, inpx)
                totcyc_loss = ycyc_loss + xcyc_loss
                
                IDX_loss = 0.5*cycle_loss(Xsame, inpx)
                IDY_loss = 0.5*cycle_loss(Ysame, inpy)
                
                tot_genxy_loss = genxy_loss + totcyc_loss + IDY_loss
                tot_genyx_loss = genyx_loss + totcyc_loss + IDX_loss
                
                disy_loss,_,_ = discriminator_loss(Yreal, X2Y)
                disx_loss,_,_ = discriminator_loss(Xreal, Y2X)
                
            
            disy_grads = tape.gradient(disy_loss, disy.trainable_variables)
            disx_grads = tape.gradient(disx_loss, disx.trainable_variables)
            dxOpt.apply_gradients(zip(disx_grads, disx.trainable_variables))
            dyOpt.apply_gradients(zip(disy_grads, disy.trainable_variables))
            
            if (n%1)==0:
                genxy_grads = tape.gradient(tot_genxy_loss, genxy.trainable_variables)
                genyx_grads = tape.gradient(tot_genyx_loss, genyx.trainable_variables)
                gxyOpt.apply_gradients(zip(genxy_grads, genxy.trainable_variables))
                gyxOpt.apply_gradients(zip(genyx_grads, genyx.trainable_variables))
            
            end_step = time.time()
            sys.stdout.write("\rEpoch: %d; Step: %d/%d; Batch loss (GXY/GYX/DY/DX/IDY/IDX/CYC): %6.3f/%6.3f/%6.3f/%6.3f/%6.3f/%6.3f/%6.3f; Time elapsed: %.1f s"%(
                                m+1, n+1, steps, genxy_loss.numpy(), genyx_loss.numpy(), disy_loss.numpy(), disx_loss.numpy(), IDY_loss, IDX_loss, totcyc_loss.numpy(), end_step-start_step))
            
            tots[0]+=genxy_loss.numpy()*amt;tots[1]+=genyx_loss.numpy()*amt
            tots[2]+=disy_loss.numpy()*amt;tots[3]+=disx_loss.numpy()*amt
            tots[4]+=IDY_loss.numpy()*amt;tots[5]+=IDX_loss.numpy()*amt
            tots[6]+=totcyc_loss.numpy()*amt
        
        if svwts:
            genxy.save_weights('genxy.wts')
            genyx.save_weights('genyx.wts')
            disx.save_weights('disx.wts')
            disy.save_weights('disy.wts')
        
        generate_images(AUT[0:1], SUM[0:1])
            
        end_epoch = time.time()
        sys.stdout.write("\rEpoch: %d; Total loss (GXY/GYX/DY/DX/IDY/IDX/CYC): %6.3f/%6.3f/%6.3f/%6.3f/%6.3f/%6.3f/%6.3f; Time elapsed: %.1f s\n"%(
                            m+1, tots[0]/steps, tots[1]/steps, tots[2]/steps,
                            tots[3]/steps, tots[4]/steps, tots[5]/steps,
                            tots[6]/steps, end_epoch-start_epoch))

def generate_images(fall, summer, fn='image.jpg'):
    if len(fall.shape)==3:
        fall = fall[tf.newaxis, ...]
    f2s = genxy(fall, training=True)[0]*0.5+0.5
    fall = np.squeeze(fall) if len(fall.shape)==4 else fall
    fall = fall*0.5+0.5
    
    if len(summer.shape)==3:
        summer = summer[tf.newaxis, ...]
    s2f = genyx(summer, training=True)[0]*0.5+0.5
    summer = np.squeeze(summer) if len(summer.shape)==4 else summer
    summer = summer*0.5+0.5
    
    fig = plt.figure(figsize=(15,15))
    
    ax1 = plt.subplot(221)
    ax1.set_title("Real")
    ax1.imshow(fall)
    ax1.set_axis_off()
    
    ax2 = plt.subplot(222)
    ax2.set_title("Generated")
    ax2.imshow(f2s)
    ax2.set_axis_off()
    
    ax3 = plt.subplot(223)
    ax3.imshow(summer)
    ax3.set_axis_off()
    
    ax4 = plt.subplot(224)
    ax4.imshow(s2f)
    ax4.set_axis_off()
    
    fig.savefig('/mnt/c/Users/joell/Desktop/'+fn, dpi=500)
    plt.close('all')

#genxy.load_weights('genxy.wts');genyx.load_weights('genyx.wts');disy.load_weights('disy.wts');disx.load_weights('disx.wts')
#train((AUT,SUM), lr=2e-4, epochs=15)
#genxy.save_weights('genxy.wts');genyx.save_weights('genyx.wts');disx.save_weights('disx.wts');disy.save_weights('disy.wts')
