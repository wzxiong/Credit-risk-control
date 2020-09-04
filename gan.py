from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import os
from tensorflow.keras import layers

from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import time
from scipy import signal
print(tf.__version__)


df = pd.read_csv('./data/train.csv')
df = df[df.label==0]
import_list = []
with open('./data/train_features.txt') as f:
    for line in f:
        import_list.append(line.rstrip())

print(df[import_list].describe().loc['mean'])
#trainECG = df[import_list].fillna(0).iloc[:,:].values
ss = StandardScaler()
trainECG = ss.fit_transform(df.fillna(0)[import_list])
trainECG = np.expand_dims(trainECG,2)

BUFFER_SIZE = 600
BATCH_SIZE = 128
inputLength = 55
model_dir='./model/'

train_dataset = tf.data.Dataset.from_tensor_slices(trainECG).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def build_generator():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(inputLength*BATCH_SIZE, input_shape=(inputLength,1),dtype='float32'))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 1,kernel_size = 5, activation = 'tanh', padding = 'same'))

    return generator

def build_discriminator():

    discriminator = tf.keras.Sequential()
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same',input_shape=(inputLength,1)))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Flatten(input_shape=(inputLength,1)))
    discriminator.add(layers.Dense(64,dtype='float32'))
    discriminator.add(layers.Dropout(0.4))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dense(1, activation='tanh',dtype='float32'))

    return discriminator

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(-0.9*tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(0.9*tf.ones_like(fake_output), fake_output)
    total_loss = 0.4*real_loss+0.6*fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(-0.9*tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

noise = tf.random.normal([20,55,1])
generateECG = generator(noise)
discriminator(generateECG)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

generator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
total_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

EPOCHS = 20
noise_dim = inputLength
num_examples_to_generate = 1000
seed = tf.random.normal([num_examples_to_generate, noise_dim, 1],dtype='float32')
seed = tf.cos(4*seed)


@tf.function
def train_step(ECG):
    noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
    noise = tf.cos(4*noise)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as total_tape:
      generated_ECG = generator(noise, training=True)

      real_output = discriminator(ECG, training=True)
      fake_output = discriminator(generated_ECG, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      total_loss = tf.tanh(tf.abs(gen_loss)-tf.abs(disc_loss))
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_total = total_tape.gradient(total_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    total_optimizer.apply_gradients(zip(gradients_of_total,generator.trainable_variables))
    return gen_loss,disc_loss,total_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        cunt = 0
        for image_batch in dataset:
            cunt+=1
            gen_loss,disc_loss,total_loss = train_step(image_batch)
            if cunt%100==0:
                generate_and_save_images(generator, epoch + 1, seed)
                print ('gen loss {} __ dis loss {} __tol loss{}'.format(gen_loss, disc_loss, total_loss))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        predictions = generate_and_save_images(generator,epochs, seed)
        predictions = np.array(predictions).squeeze()
        predictions = ss.inverse_transform(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.columns = import_list
        print(process(predictions).describe().loc['mean'])
        #pd.DataFrame(predictions).to_csv('./data/prediction'+str(epoch)+'.csv',index=None)
        generator.save('./model/model_good'+str(epoch)+'.h5')
    predictions = generate_and_save_images(generator,epochs, seed)
    return predictions

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    return predictions

def process(data):
    for key in import_list:
        data[key] = [i if i > 0 else np.nan for i in data[key]]
    return data

predictions = train(train_dataset, 10)
#predictions = np.array(predictions).squeeze()
#predictions = ss.inverse_transform(predictions)
#print(pd.DataFrame(predictions).describe())
#pd.DataFrame(predictions).to_csv('./data/prediction.csv',index=None)

print('load model')
model = keras.models.load_model('./model/model_good5.h5')

total_num = 1000000
times = int(total_num / 1000)

data = pd.DataFrame()
for _ in range(times):
    seed = tf.random.normal([num_examples_to_generate, noise_dim, 1],dtype='float32')
    seed = tf.cos(4*seed)
    predictions = model(seed, training=False)
    predictions = np.array(predictions).squeeze()
    predictions = ss.inverse_transform(predictions)
    data = pd.concat([data,pd.DataFrame(predictions)])
data.columns = import_list
data = process(data)
print(data.describe().loc['mean'])
data.to_csv('./data/prediction_good.csv',index=None)
