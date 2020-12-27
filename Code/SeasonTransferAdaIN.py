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
LATENT_DIM = 64

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

class AdaIN(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def build(self, inp):
        self.fc = tf.keras.layers.Dense(2*inp[-1])
    
    def call(self, inp, z):
        fc = self.fc(z)
        gamma = fc[:, :inp.shape[-1]]
        beta = fc[:, inp.shape[-1]:]
        
        mean = tf.reduce_mean(inp, axis=[0,1,2])
        mean = mean[tf.newaxis, tf.newaxis, tf.newaxis, :]
        
        im = inp-mean
        
        sig = tf.math.reduce_std(im, axis=[0,1,2])
        sig = sig[tf.newaxis, tf.newaxis, tf.newaxis, :]
        out = im/(sig+1e-8)
        out = tf.multiply(gamma[:, tf.newaxis, tf.newaxis, :], out) + beta[:, tf.newaxis, tf.newaxis, :]
        return out

class downsample(tf.keras.Model):
    def __init__(self, filters, size, apply_norm=0):
        super(downsample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=2, padding='same',
                                           kernel_initializer=initializer, use_bias=False)
        self.norm = apply_norm
        if apply_norm==0:
            self.AI = AdaIN()
        elif apply_norm==1:
            self.AI = tf.keras.layers.BatchNormalization()
        else:
            self.AI = None
        self.relu = tf.keras.layers.ReLU()
    
    def __call__(self, inp, z=None):
        out = self.conv(inp)
        if self.norm==0:
            out = self.AI(out, z)
        elif self.norm==1:
            out = self.AI(out)
        out = self.relu(out)
        return out

class upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(upsample, self).__init__()
        self.deconv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=size, strides=2, padding='same',
                                                      kernel_initializer=initializer, use_bias=False)
        self.AI = AdaIN()
        self.drop = apply_dropout
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.keras.layers.ReLU()
    
    def __call__(self, inp, z):
        out = self.deconv(inp)
        out = self.AI(out, z)
        if self.drop:
            out = self.dropout(out)
        out = self.relu(out)
        return out

def Generator():
    inim = tf.keras.layers.Input(shape=dims) # bs, 256, 256, 3
    inz = tf.keras.layers.Input(shape=(LATENT_DIM,))
    
    downstack = [
        downsample(64, 4, apply_norm=3), # bs, 128, 128, 64
        downsample(128, 4, 0), # bs, 64, 64, 128
        downsample(256, 4, 0), # bs, 32, 32, 256
        downsample(512, 4, 0), # bs, 16, 16, 512
        downsample(512, 4, 0), # bs, 8, 8, 512
        downsample(512, 4, 0), # bs, 4, 4, 512
        downsample(512, 4, 0), # bs, 2, 2, 512
        downsample(512, 4, 0) # bs, 1, 1, 512
        ]
    
    upstack = [
        upsample(512, 4, apply_dropout=True), # bs, 2, 2, 1024
        upsample(512, 4, apply_dropout=True), # bs, 4, 4, 1024
        upsample(512, 4, apply_dropout=True), # bs, 8, 8, 1024
        upsample(512, 4), # bs, 16, 16, 1024
        upsample(256, 4), # bs, 32, 32, 512
        upsample(128, 4), # bs, 64, 64, 256
        upsample(64, 4) # bs, 128, 128, 128
        ]
    
    last = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, kernel_initializer=initializer,
                                  padding='same', activation='tanh')
    
    # operations
    out = inim
    Z = inz
    
    skips = []
    for layer in downstack:
        out = layer(out, Z)
        skips.append(out)
    
    for count, layer in enumerate(upstack):
        out = layer(out, Z)
        out = tf.keras.layers.Concatenate()([out, skips[-(count+2)]])
    
    out = last(out)
    
    return tf.keras.Model(inputs=[inim, inz], outputs=out)

def Discriminator():
    inp = tf.keras.layers.Input(shape=dims) # bs, 256, 256, 3
    
    out = downsample(64, 4, 3)(inp) # bs, 128, 128, 64
    out = downsample(128, 4, 1)(out) # bs, 64, 64, 128
    out = downsample(256, 4, 1)(out) # bs, 32, 32, 256
    
    out = tf.keras.layers.ZeroPadding2D((1,1))(out) # bs, 34, 34, 256
    
    out = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding='valid',
                           kernel_initializer=initializer)(out) # bs, 31, 31, 512
    
    out = tf.keras.layers.BatchNormalization()(out)
    
    out = tf.keras.layers.LeakyReLU(0.2)(out)
    
    out = tf.keras.layers.ZeroPadding2D()(out) # bs, 33, 33, 512
    
    out = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='valid',
                                 kernel_initializer=initializer)(out) # bs, 30, 30, 1
    
    return tf.keras.Model(inputs=inp, outputs=out)

#pip install -q git+https://github.com/tensorflow/examples.git
#from tensorflow_examples.models.pix2pix import pix2pix

genxy = Generator()
genyx = Generator()
disx = Discriminator()
disy = Discriminator()

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
                Zx = tf.random.normal((BATCH_SIZE, LATENT_DIM))
                Ys= genxy([inpx, Zx], training=True)
                Xcyc = genyx([Ys, Zx], training=True)
                Zy = tf.random.normal((BATCH_SIZE, LATENT_DIM))
                Xs = genyx([inpy, Zy], training=True)
                Ycyc = genxy([Xs, Zy], training=True)
                Ysame = genxy([inpy, Zx], training=True)
                Xsame = genyx([inpx, Zy], training=True)
                
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
    Zfall = tf.random.normal((1,LATENT_DIM))
    f2s = genxy([fall, Zfall], training=True)[0]*0.5+0.5
    fall = np.squeeze(fall) if len(fall.shape)==4 else fall
    fall = fall*0.5+0.5
    
    if len(summer.shape)==3:
        summer = summer[tf.newaxis, ...]
    Zsummer = tf.random.normal((1, LATENT_DIM))
    s2f = genyx([summer, Zsummer], training=True)[0]*0.5+0.5
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
train((AUT,SUM), lr=2e-4, epochs=15, svwts=True)
#genxy.save_weights('genxy.wts');genyx.save_weights('genyx.wts');disx.save_weights('disx.wts');disy.save_weights('disy.wts')
