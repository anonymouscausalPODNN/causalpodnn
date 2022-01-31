import warnings
warnings.filterwarnings("ignore")
import os
import psutil
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import cv2
import podnn_tensorflow_train
tf.get_logger().setLevel('INFO')
from scipy.spatial.distance import  pdist
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
process = psutil.Process(os.getpid())

seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)


im_height = 32
im_width = 32

# Parameters
severity = 5   # the severity of transformations
n_models = 6   # number of models which is the equivaent of number of transformations to be inverted
n_transforms = n_models
epochs = 6     # number of epochs for training the models
small_sample = 4200     # samples sample size for quick comparision (for corresponding result paper, use the entire dataset. Also set the n_models to 10)
batch_size = n_models*7
n_iterations = batch_size*epochs


unit_model = [
    tf.keras.layers.Conv2D(8, 3, input_shape=[im_height,im_width,1],padding='SAME')
    #tf.keras.layers.BatchNormalization(axis=0)
    #tf.keras.layers.MaxPool2D((2,2)),
    #tf.keras.layers.Dropout(rate=0.2)

]
unit_model_2 = [
    tf.keras.layers.Conv2D(8, 3, input_shape=[im_height,im_width,8],padding='SAME')
    #tf.keras.layers.BatchNormalization(axis=0)
    #tf.keras.layers.Dropout(rate=0.5)
    #tf.keras.layers.MaxPooling2D(2),
]
unit_model_3 = [
    tf.keras.layers.Conv2D(8, 3, input_shape=[im_height,im_width,8],padding='SAME')
    #tf.keras.layers.BatchNormalization(axis=0)
]
unit_model_4 = [
    tf.keras.layers.Conv2D(8, 3, input_shape=[im_height,im_width,8],padding='SAME')
    #tf.keras.layers.BatchNormalization(axis=0)
    #tf.keras.layers.MaxPooling2D(2),
]

unit_model_5 = [
    tf.keras.layers.Conv2D(1, 3, input_shape=[im_height,im_width,8],padding='SAME')
    #tf.keras.layers.MaxPooling2D(2),
]

w_init = tf.ones_initializer()
w = tf.Variable(initial_value=w_init(shape=4),trainable=True,dtype='float32')
init= tf.constant_initializer(np.ones(4))
orthparam = [
    #tf.cast(w,'float64')
    #tf.keras.layers.Dense(4,kernel_initializer=init,trainable=True)
    tf.keras.layers.Dense(n_models-1,trainable=True)
    #tf.Variable(initial_value=tf.ones((1,4)),dtype=tf.float32,trainable=True)
]

class podnnModel(layers.Layer):
    # this is the base class for creating the generator models
    # orhogonalization layer can be applied after each parallelLayer. In this example it is put only after the fourth parallel layer
    # for comparing with non-orthogonal version, in the call function simply comment the OrthogonalLayer4 layer
    def __init__(self):
        super(podnnModel, self).__init__()
        pass


    def build(self,input_shape):
        self.InputLayer = podnn_tensorflow_train.InputLayer(n_models=n_models)
        self.ParallelLayer = podnn_tensorflow_train.ParallelLayer(unit_model)
        #self.OrthogonalLayer = podnn_tensorflow_train.OrthogonalLayer2D(orthparam=orthparam)
        self.ParallelLayer2 = podnn_tensorflow_train.ParallelLayer(unit_model_2)
       #self.OrthogonalLayer2 = podnn_tensorflow_train.OrthogonalLayer2D(orthparam=orthparam)
        self.ParallelLayer3 = podnn_tensorflow_train.ParallelLayer(unit_model_3)
        #self.OrthogonalLayer3 = podnn_tensorflow_train.OrthogonalLayer2D(orthparam=orthparam)
        self.ParallelLayer4 = podnn_tensorflow_train.ParallelLayer(unit_model_4)
        self.OrthogonalLayer4 = podnn_tensorflow_train.OrthogonalLayer2D(orthparam=orthparam)
        self.ParallelLayer5 = podnn_tensorflow_train.ParallelLayer(unit_model_5)

    def call(self,x):
        x = self.InputLayer(x)
        x = self.ParallelLayer(x)
        #x = self.OrthogonalLayer(x)
        x = self.ParallelLayer2(x)
        #x = self.OrthogonalLayer2(x)
        x = self.ParallelLayer3(x)
        #x = self.OrthogonalLayer3(x)
        x = self.ParallelLayer4(x)
        x = self.OrthogonalLayer4(x)
        x = self.ParallelLayer5(x)

        return x

common_layer = podnnModel()

class Generator(Model):
    # wrapper generator class around podnnModel class  for creating generators
    def __init__(self,model_num):
        super(Generator,self).__init__()
        self.model_num = model_num

    def build(self,input_dim):
        self.common_layer = common_layer

    def call(self,inputs):
        all_models = self.common_layer(inputs)
        #all_models = podnnModel()(inputs)
        current_model = all_models[self.model_num,:,:,:]

        return current_model


class Discriminator(Model):
    # this is the class for creating the discriminator
    def __init__(self):
        super(Discriminator,self).__init__()

    def build(self,input_shape):
        self.conv_0 = layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[im_height,im_width, 1])
        self.relu_0 = layers.LeakyReLU()
        #self.dropout_0 = layers.Dropout(0.3)
        self.conv_1 = layers.Conv2D(32, (3, 3), strides=(2,2), padding='same',
                                input_shape=[im_height,im_width, 1])
        self.relu_1 = layers.LeakyReLU()
        self.dropout_1 = layers.Dropout(0.3)

        self.conv_2 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.relu_2 = layers.LeakyReLU()
        self.dropout_2 = layers.Dropout(0.3)

        self.flatten =  layers.Flatten()
        self.dense1 = layers.Dense(128, activation='elu')
        self.dense2 = layers.Dense(1,activation='sigmoid')

    def call(self,inputs):
        x =  self.conv_1(inputs)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.flatten(x)
        x1 = self.dense1(x)   # When using this layer, the couldn't go beyond 4 ouf ot 5
        x2 = self.dense2(x)
        return x1,x2
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.pad(x_train, [[0, 0], [2,2], [2,2]])
x_test = np.pad(x_test, [[0, 0], [2,2], [2,2]])

x_train = x_train / 255.0
x_test = x_test / 255.0


def add_noise(x):
    n_images = x.shape[0]
    n_rows = x.shape[1]
    n_cols = x.shape[2]
    mean = 0.5
    std = 0.3

    noise = np.random.normal(mean,std,(n_images,n_rows,n_cols))
    x_noise = x+ .3*noise
    return x_noise

def inversion(x):
    x_inverted = 1 - x
    return x_inverted

def right_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,severity],[0,1,0]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def left_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,-severity],[0,1,0]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def down_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,0],[0,1,severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def up_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,0],[0,1,-severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def right_up_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,severity],[0,1,-severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def right_down_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,severity],[0,1,severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def left_up_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,-severity],[0,1,-severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def left_down_translation(x):
    x_translated = x.copy()
    M = np.float32([[1,0,-severity],[0,1,severity]])
    for i in range(len(x)):
        x_translated[i,:,:] = cv2.warpAffine(x[i,:,:],M,(im_height,im_width))
    return x_translated

def rotation(x):
    x_rotated = x.copy()
    for i in range(len(x)):
        x_rotated[i,:,:] = cv2.rotate(x[i,:,:],cv2.ROTATE_90_CLOCKWISE)
    return x_rotated

def data_podnn_prepare(x):
    x = np.expand_dims(x, axis=3)

    x_list = np.array([x] * n_models)

    x_list = np.transpose(x_list, [1, 0, 2, 3, 4])
    return x_list

def dataset_formation(x_train,n_transforms_local):
    # this function creates transformed data points
    cut_point = int(len(x_train)/2)
    x_before_transform = x_train[:cut_point]
    x_distribution = x_train[cut_point:]
    x_inverted = inversion(x_before_transform)
    x_noise = add_noise(x_before_transform)
    x_right_translated = right_translation(x_before_transform)
    x_left_translated = left_translation(x_before_transform)
    x_up_translated = up_translation(x_before_transform)
    x_down_translated = down_translation(x_before_transform)
    x_right_up_translated = right_up_translation(x_before_transform)
    x_right_down_translated = right_down_translation(x_before_transform)
    x_left_up_translated = left_up_translation(x_before_transform)
    x_left_down_translated = left_down_translation(x_before_transform)
    x_transformed = np.concatenate((x_inverted,x_noise,x_right_translated,x_left_translated,x_up_translated,x_down_translated,\
            x_right_up_translated,x_right_down_translated,x_left_up_translated,x_left_down_translated),axis=0)

    x_transformed = x_transformed[:n_transforms_local*cut_point]
    x_distribution = np.repeat(x_distribution,n_transforms_local,axis=0)
    labels = list(np.arange(n_transforms_local))
    labels = np.repeat(labels,cut_point)
    p = np.random.permutation(len(x_transformed))

    counter = 0
    for i in range(cut_point):
        for j in range(n_transforms_local):
            start = j*cut_point
            p[counter] = start + i
            counter += 1
    x_transformed = x_transformed[p]
    labels =  labels[p]

    p = np.random.permutation(len(x_distribution))
    x_distribution = x_distribution[p]

    return_list = [x_distribution,x_transformed,x_inverted,x_noise,x_right_translated,x_left_translated,x_up_translated,x_down_translated,\
            x_right_up_translated,x_right_down_translated,x_left_up_translated,x_left_down_translated]
    return return_list[:n_transforms_local+2],labels

# list of transformations
trans_list = ['x_inverted','x_noise','x_right_translated','x_left_translated','x_up_translated','x_down_translated',\
            'x_right_up_translated','x_right_down_translated','x_left_up_translated','x_left_down_translated']


task_list = trans_list
data,labels  = dataset_formation(x_train,n_models)
x_transformed = data[1]
x_distribution = data[0]


x = []
for i in range(len(data)):
    x.append(np.expand_dims(data[i],axis=3))

data_tuple = [d[:small_sample] for d in x]
data_tuple.append(labels[:small_sample])
data_tuple = tuple(data_tuple)

train_ds = tf.data.Dataset.from_tensor_slices(
        (data_tuple)).batch(batch_size)


generators = []
for i in range(n_models):
    generators.append(Generator(model_num=i))
discriminator = Discriminator()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss_against(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 1*real_loss + fake_loss
    return total_loss

def discriminator_loss_log(real_output,fake_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return fake_loss

def discriminator_loss_real(real_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return real_loss

def discriminator_loss(real_output, fake_output):

    fake_loss = 0
    counter = n_transforms
    for i in range(n_transforms):
        if len(fake_output[i]) == 0:
            counter -= 1
        else:
            fake_loss += tf.reduce_mean(tf.math.log(1 - fake_output[i]))
    fake_loss /= counter
    total_loss = tf.reduce_mean(tf.math.log(real_output)) +  fake_loss
    total_loss = -total_loss
    return total_loss

def generator_loss(generator,fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generator_identity_loss(y_true,y_pred):
    return tf.reduce_mean(tf.losses.mse(y_true,y_pred))

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#-------------------------------
#@tf.function
def train_identity(x_noise):
    # this function is used when one wants to first make generators to learn identity
    with tf.GradientTape(persistent=True) as gen_tape:

        generated_imagess = []
        for i in range(n_transforms):
            generated_imagess.append(generators[i](x_noise, training=True))

        gen_losses = []
        for i in range(n_transforms):
            gen_losses.append( generator_identity_loss(generated_imagess[i], x_noise) )

    gradients_generators = []
    for i in range(n_transforms):
         gradients_generators.append( gen_tape.gradient(gen_losses[i], generators[i].trainable_variables) )


    for i in range(n_transforms):
        generator_optimizer.apply_gradients(zip(gradients_generators[i], generators[i].trainable_variables))

    return gen_losses


def find_max(x):
    max = -1000
    max_idx = -1
    for i in range(len(x)):
        if np.isreal(x[i]):
            if x[i]>max:
                max = x[i]
                max_idx = i

    return max_idx


def find_min(x):
    min = 1000
    min_idx = -1
    for i in range(len(x)):
        if np.isreal(x[i]):
            if x[i] < min:
                min = x[i]
                min_idx = i

    return min_idx


#@tf.function
def train_step(x_noise, x_intact,min_prob):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

        x_noise = np.array(x_noise)
        x_intact = np.array(x_intact)
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        generated_imagess = []
        fake_outputs = []
        for i in range(n_transforms):
            generated_imagess.append(generators[i](x_noise, training=False))
            fake_outputs.append(discriminator(generated_imagess[i], training=False)[1])

        real_output = discriminator(x_intact, training=False)[1]

        d_list = []
        disc_losses = [[] for _ in range(n_transforms)]
        for j in range(n_transforms):
            for i in range(len(fake_outputs[0])):
                disc_losses[j].append(discriminator_loss_against(real_output, fake_outputs[j][i]))

        for i in range(n_transforms):
            d_list.append(tf.convert_to_tensor(disc_losses[i]))

        d = tf.stack((d_list), axis=1)
        winner = tf.argmax(d, axis=1)
        winners = []
        for i in range(n_transforms):
            winners.append(np.array(winner == i))

        for i in range(n_transforms):
            print('generator_' + str(i) + '  wins:  ' + str(np.sum(winners[i])))


        woned_samples_list = []
        for i in range(n_transforms):
            expert_samples = np.where(winner==i)[0]
            idx = np.argsort(-np.array(d)[expert_samples,i])
            expert_samples = expert_samples[idx]
            woned_samples_list.append(expert_samples)


        idx_list = []
        for i in range(n_transforms):
            idx_list.append(np.mod(np.where(winners[i] == True)[0], n_transforms))

        winners_nsamples = []
        for i in range(n_transforms):
            try:
                winners_nsamples.append(np.max(np.bincount(idx_list[i])))
            except:
                winners_nsamples.append(0)
                temp = 1
            task_idx = np.argsort(-np.bincount(idx_list[i]))
            specialized_task = [task_list[i] for i in task_idx[:2]]
            print('idx' + str(i) + str(np.bincount(idx_list[i], minlength=n_transforms)) \
                  + str(specialized_task))

        #=============================Discriminator part==============================

        generated_images = [] # generated images for corresponing winners' experts
        generated_images_against = [[] for i in range(n_transforms)]
        for i in range(n_transforms):
            generated_images.append(generators[i](x_noise[winners[i], :, :, :], training=False))
            for j in range(n_transforms):
                if j==i:
                    generated_images_against[i].append([])
                if j!=i:
                    try:
                        generated_images_against[i].append(generators[j](x_noise[winners[i], :, :, :], training=False))
                    except:
                        temp = 1

        real_output = discriminator(x_intact, training=True)[1]
        fake_outputs = []   # this if for training the discriminator
        fake_outputs_against = [[] for  i in range(n_transforms)]   # this if for training the discriminator against loosing experts
        dist_feats = [[] for  i in range(n_transforms)]
        for i in range(n_transforms):
            x1,x2 = discriminator(generated_images[i], training=True)
            if len(x1)!=0:
                dist_feats[i] = x1
            fake_outputs.append(x2)
            for j in range(n_transforms):
                if j==i:
                    fake_outputs_against[i].append([])
                if j!=i:
                    fake_outputs_against[i].append(discriminator(generated_images_against[i][j], training=True)[1])


#        gen_losses = []
        disc_losses_against = [[] for i in range(n_transforms)]
        for i in range(n_transforms):
            #gen_losses.append(generator_loss(generators[i], fake_outputs_total[i]))
            for j in range(n_transforms):
                if j!=i:
                    disc_losses_against[i].append(discriminator_loss_against(real_output, fake_outputs_against[i][j]))

        disc_loss = discriminator_loss(real_output, fake_outputs)
        disc_loss_against = []
        for i in range(n_transforms):
            disc_loss_against.append(tf.reduce_mean(disc_losses_against[i])/(n_transforms-1))

        #---------------------Time to update discriminator-----------------------
        gradients_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_discriminator_against = []
        for i in range(n_transforms):
            gradients_discriminator_against.append(
                disc_tape.gradient(disc_loss_against[i], discriminator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
        for i in range(n_transforms):
            discriminator_optimizer.apply_gradients(
                zip(gradients_discriminator_against[i], discriminator.trainable_variables))

        #===========================Generate new images====================================

        #--------------------------Mutation--------------------------------------
        mutation = True
        if mutation == True:

            std_list = []
            for i in range(n_transforms):
                if len(dist_feats[i]) >1:
                    dist_feats[i] = np.array(dist_feats[i])
                    dists = pdist(dist_feats[i])
                    std_list.append(np.std(dists))
                    print('expert:'+str(i) + '   std:'+str(std_list[-1]))
                else:
                    std_list.append([])

            std_list = np.array(std_list)


            lost_experts = [i for i,e in enumerate(idx_list) if len(e)==0]
            won_experts = [i for i, e in enumerate(idx_list) if len(e) > 1]
            if len(lost_experts)!=0:
                r_lost = np.random.randint(len(lost_experts))
                r_won_max = find_max(std_list)
                r_won_min = find_min(std_list)
                r = np.random.rand()
                min_prob=0.5
                if r < min_prob:
                    r_won = r_won_max
                    print('max selected')
                else:
                    r_won = r_won_min
                    print('min selected')

                print('from expert:' + str(r_won) +'    to expert:'+str(lost_experts[r_lost]))
              #  r_won = np.random.randint(len(won_experts))

                #idx = np.where(winners[won_experts[r_won]]==True)[0]
                n_samples = tf.cast(tf.math.ceil(tf.cast(len(idx), 'float32') * tf.constant(.5)), 'int32')
                #r_select = np.random.randint(len(idx),size=n_samples)
                r_select = woned_samples_list[r_won][-n_samples:]
                winners[lost_experts[r_lost]][r_select] = True
                winners[r_won][r_select] = False

        # ---------------------------Generators' losses for winner experts--------------------------------------
        generated_images = [] # generated images for corresponing winners' experts
        for i in range(n_transforms):
            generated_images.append(generators[i](x_noise[winners[i], :, :, :], training=True))


        fake_outputs = []   # this if for training the discriminator
        for i in range(n_transforms):
            fake_outputs.append(discriminator(generated_images[i], training=False)[1])

        gen_losses = []
        for i in range(n_transforms):
            gen_losses.append(generator_loss(generators[i], fake_outputs[i]))

        #---------------------------Time to update generators---------------------------------
        gradients_generators = []
        for i in range(n_transforms):
            gradients_generators.append(gen_tape.gradient(gen_losses[i], generators[i].trainable_variables))

        for i in range(n_transforms):
            generator_optimizer.apply_gradients(zip(gradients_generators[i], generators[i].trainable_variables))



    return gen_losses, disc_loss, winners_nsamples, min_prob
#--------------------------------
d0_inv_list = []
d1_inv_list = []
d2_inv_list = []
d0_trans_list = []
d1_trans_list = []
d2_trans_list = []
d0_noise_list = []
d1_noise_list = []
d2_noise_list = []



train_identity_flag = 0
identity_epochs = 2
if train_identity_flag==1:

    for epoch in range(identity_epochs):

        for train_contents in train_ds:

            gen_loss_list = \
                train_identity(train_contents[1])

            for i in range(n_models):
                print('epoch:' + str(epoch) + '  Generator loss' + str(i) + ': ' + str(gen_loss_list[i]))
            print('---------------------------------------------------------')


def disc_score(X,X_real):

    generated_images = []
    for i in range(n_transforms):
        generated_images.append( generators[i](X, training=False) )

    real_output = discriminator(X_real, training=False)[1]

    fake_outputs = []
    for i in range(n_transforms):
        fake_outputs.append( discriminator(generated_images[i], training=False)[1] )

    disc_losses = []
    for i in range(n_transforms):
         disc_losses.append(  discriminator_loss_log(real_output, fake_outputs[i]) )

    return disc_losses


disc_loss_list = [ []  for i in range(n_transforms)]
winners_nsamples_list = []
orth_weights_list = []


# main training loop
iteration = 0
min_prob = 1.0
train_disc = True
if train_disc == True:

    for epoch in range(epochs):

          for train_contents in train_ds:
              print('memory:'+str(process.memory_percent()))
              gen_loss_list,disc_loss, winners_nsamples, min_prob = \
                  train_step(train_contents[1],train_contents[0],min_prob)
              winners_nsamples_list.append(np.array(winners_nsamples))
              iteration += 1

              for i in range(n_models):
                    print('epoch:'+str(iteration)+  'min_prob:'+str(min_prob),  '  Generator loss'+str(i)+': ' + str(gen_loss_list[i]))
              print(' Discrimintator loss:' + str(disc_loss))
              print('---------------------------------------------------------')



def visualize(x,filter_num):

    disc_list = []

    for i in range(2, n_transforms + 2):
        disc_list.append(disc_score(x[i], x[0]))
        print('to check:'+str(np.array(disc_score(x[i], x[0]))))

    disc_array = np.array(disc_list)
    max_trans = np.argmax(disc_array,axis=0)

    for i in range(n_transforms):
        disc_list.append( disc_score(tf.expand_dims(x[1][n_models+i],axis=0),tf.expand_dims(x[0][n_models+i],axis=0) ))

    fig, ax = plt.subplots(n_transforms + 1, n_transforms)

    for i in range(n_transforms):
        ax[0,i].imshow(x[1][i,:,:,0],cmap='gray')

    for i in range(n_transforms):
        x_generated = generators[i](x[1])
        for j in range(n_transforms):
            ax[i+1, j].imshow(x_generated[j, :, :, 0], cmap='gray')
            if j==0:
                ax[i+1,j].set_ylabel(trans_list[max_trans[i]],rotation=45)


    plt.show()

    fig, ax = plt.subplots(n_transforms+1,n_transforms)

    for i in range(n_transforms):
        ax[0,i].imshow(x[1][i,:,:,0],cmap='gray')

    for i in range(n_transforms):
        x1 = generators[i]._layers[0]._layers[0](x[1])
        x2 = generators[i]._layers[0]._layers[1](x1)
        x3 = generators[i]._layers[0]._layers[2](x2)
        x4 = generators[i]._layers[0]._layers[3](x3)
        x5 = generators[i]._layers[0]._layers[4](x4)

        print('dot product:'+str(tf.reduce_sum(tf.multiply(x3[0,0,:,:,0],x3[1,0,:,:,0]))))

        for j in range(n_transforms):
            ax[i+1,j].imshow(x3[i,j,:,:,filter_num],cmap='gray')
            if j==0:
                ax[i + 1, j].set_ylabel(trans_list[max_trans[i]], rotation=45)

    fig, ax = plt.subplots(8 , n_transforms)
    ax[0,0].set_title('filter of parallel layer 1')
    for i in range(8):
        for j in range(n_transforms):
            if i==0:
                ax[i,j].set_title('parallel model '+str(j))
            if j==0:
                ax[i, j].set_ylabel('filter ' + str(i),rotation=90)
            ax[i,j].imshow(generators[0]._layers[0]._layers[1].weights[j*2][:,:,0,i],cmap='gray')

    fig, ax = plt.subplots(8 , n_transforms)
    ax[0,0].set_title('filter of parallel layer 4')
    for i in range(8):
        for j in range(n_transforms):
            if i==0:
                ax[i,j].set_title('parallel model '+str(j))
            if j==0:
                ax[i, j].set_ylabel('filter ' + str(i),rotation=90)
            ax[i,j].imshow(generators[0]._layers[0]._layers[3].weights[j*2][:,:,0,i],cmap='gray')


    #plt.show()

    return x1

temp = visualize(x,filter_num=0)

fig, axs = plt.subplots(6,2,figsize=(20,15))
fig.subplots_adjust(hspace=0.5)
axs = axs.ravel()

epochs = len(disc_loss_list[0])
fix_idx = np.arange(epochs)
for i in range(n_transforms):
    current_task_disc = np.array(disc_loss_list[i])
    for j in range(n_transforms):
        #idx = np.where(np.mod(fix_idx,n_transforms)==j)
        axs[i].plot(current_task_disc[:,j])
    axs[i].set_title(task_list[i])

winners_nsamples_list = np.array(winners_nsamples_list)
for i in range(n_transforms):
    axs[-1].plot(winners_nsamples_list[:,i])
axs[-1].set_title('Number of won samples per tasks')

plt.show()



