"""ControlGAN for CIFAR-10"""
"""Most of the codes are from https://github.com/igul222/improved_wgan_training"""

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf

import time
import functools
import locale

from keras.preprocessing.image import ImageDataGenerator

locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './cifar10'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
NORMALIZATION_C = True # Use batchnorm (or layernorm) in classifier?
ORTHO_REG = False
CT_REG = True
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 1 # Critic steps per generator steps
INCEPTION_FREQUENCY = 5000 # How frequently to calculate Inception score
STOP_ACC_CLASS = 1.0

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1.0 # How to scale generator's ACGAN loss relative to WGAN loss
INI_GAMMA = 0.0 #Initial gamma

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Ortho_reg(w):
    w_shape = w.shape.as_list()
    w_ = tf.reshape(w, [-1, w_shape[-1]])
    w_2 = tf.matmul(tf.transpose(w_), w_)
    frob_norm = tf.norm(tf.multiply(w_2, (tf.ones([w_shape[-1],w_shape[-1]]) - tf.eye(w_shape[-1]))))
    frob_norm = tf.reshape(frob_norm, [-1])
    return tf.reshape(tf.square(frob_norm), [])

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None
    if CONDITIONAL and ACGAN and ('Classifier' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Classifier' in name) and NORMALIZATION_C:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)            
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity(output)            
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def OptimizedResBlockClass1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=32)
    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=32, output_dim=32)
    conv_shortcut = lib.ops.conv2d.Conv2D
    shortcut = conv_shortcut('Classifier.1.Shortcut', input_dim=3, output_dim=32, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Classifier.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity(output)            
    output = conv_2('Classifier.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels, kp=0.5):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    if CT_REG and kp!=1.0:
        output = tf.nn.dropout(output, 0.8)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    if CT_REG:
        output = tf.nn.dropout(output, kp)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    if CT_REG:
        output = tf.nn.dropout(output, kp)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    return output_wgan

def Classifier(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockClass1(output)
    output = ResidualBlock('Classifier.2', 32, 32, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.3', 32, 32, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.4', 32, 128, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Classifier.5', 128, 128, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.6', 128, 128, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.7', 128, 512, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Classifier.8', 512, 512, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.9', 512, 512, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_cgan = lib.ops.linear.Linear('Classifier.Output', 512, 10, output)
    return output_cgan

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    gamma_input = tf.placeholder(tf.float32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data_splits.append(Generator(int(BATCH_SIZE/len(DEVICES)), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:int(len(DEVICES)/2)]
    DEVICES_A = DEVICES[int(len(DEVICES)/2):]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_fake_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i], 
                all_real_data_splits[len(DEVICES_A)+i], 
                fake_data_splits[i], 
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_all_acgan = Classifier(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
            disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
            #disc_costs.append(tf.reduce_mean(tf.losses.hinge_loss(labels = tf.concat([tf.ones([BATCH_SIZE]),tf.zeros([BATCH_SIZE])], 0), logits = disc_all)))
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:int(BATCH_SIZE/len(DEVICES_A))], labels=real_and_fake_labels[:int(BATCH_SIZE/len(DEVICES_A))])
                ))
                disc_acgan_fake_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[int(BATCH_SIZE/len(DEVICES_A)):], labels=real_and_fake_labels[int(BATCH_SIZE/len(DEVICES_A)):])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:int(BATCH_SIZE/len(DEVICES_A))], dimension=1)),
                            real_and_fake_labels[:int(BATCH_SIZE/len(DEVICES_A))]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[int(BATCH_SIZE/len(DEVICES_A)):], dimension=1)),
                            real_and_fake_labels[int(BATCH_SIZE/len(DEVICES_A)):]
                        ),
                        tf.float32
                    )
                ))


    for i, device in enumerate(DEVICES_B):
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[int(BATCH_SIZE/len(DEVICES_A)),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels), [interpolates])
            if CT_REG:
                CT_D1 = Discriminator(interpolates, labels)
                CT_D2 = Discriminator(interpolates, labels)
                CT_dist = tf.maximum(.0, tf.sqrt(tf.square(CT_D1-CT_D2)))
            
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            if CT_REG:
                gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2) + 2*tf.reduce_mean(CT_dist)
            else:
                gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_fake = tf.add_n(disc_acgan_fake_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan
        class_cost = ACGAN_SCALE*disc_acgan
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')
    class_params = lib.params_with_name('Classifier.')
    gen_params = lib.params_with_name('Generator')
    if ORTHO_REG:
        ortho_reg_disc = 0.
        for param_disc in disc_params:
            disc_w = param_disc
            if ('Filters' in disc_w.name) & ('Conv' in disc_w.name):
                ortho_reg_disc += Ortho_reg(disc_w)
                print('Ortho regularized: ' + disc_w.name)

        ortho_reg_gen = 0.
        for param_gen in gen_params:
            gen_w = param_gen
            if ('Filters' in gen_w.name) & ('Conv' in gen_w.name):
                ortho_reg_gen += Ortho_reg(gen_w)
                print('Ortho regularized: ' + gen_w.name)

        ortho_reg_class = 0.
        for param_class in class_params:
            class_w = param_class
            if ('Filters' in class_w.name) & ('Conv' in class_w.name):
                ortho_reg_class += Ortho_reg(class_w)
                print('Ortho regularized: ' + class_w.name)
    
    if DECAY:
        #decay = tf.maximum(1., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
        decay = tf.cond(tf.cast(_iteration, tf.float32)/ITERS>0.5, lambda: tf.constant(0.5, dtype=tf.float32), lambda: tf.constant(1.0, dtype=tf.float32))
        decay2 = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.
        decay2 = 1.

    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = int(GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES))
            fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                disc_fake_acgan = Classifier(Generator(n_samples,fake_labels), fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (gamma_input*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay*4, beta1=0., beta2=0.9)
    class_opt = tf.train.AdamOptimizer(learning_rate=LR*decay2*5, beta1=0.9, beta2=0.999)
    if ORTHO_REG:
        gen_gv = gen_opt.compute_gradients(gen_cost + 1e-5*ortho_reg_gen, var_list=lib.params_with_name('Generator'))
        disc_gv = disc_opt.compute_gradients(disc_cost + 1e-5*ortho_reg_disc, var_list=disc_params)
        class_gv = class_opt.compute_gradients(class_cost + 1e-5*ortho_reg_class, var_list=class_params)
    else:
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        class_gv = class_opt.compute_gradients(class_cost, var_list=class_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)
    class_train_op = class_opt.apply_gradients(class_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'samples_{}.png'.format(frame))

    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in range(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv), ('C', class_gv)]:
        print("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print("\t{} ({})".format(v.name, shape_str))
        print("Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        ))

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    
    gamma_param = INI_GAMMA
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=1e-6,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  
        vertical_flip=False,
        data_format="channels_first" )

    _data,_labels = next(gen)
    datagen.fit(_data.reshape(-1, 3, 32, 32))
    
    dev_disc_acgan = -np.log(0.1)
    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration, gamma_input: gamma_param})

        for i in range(N_CRITIC):
            _data,_labels = next(gen)
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _, _disc_acgan, _disc_acgan_fake, _disc_acgan_acc, _disc_acgan_fake_acc, = session.run([disc_cost, disc_wgan, disc_train_op, disc_acgan, disc_acgan_fake, disc_acgan_acc, disc_acgan_fake_acc], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration, gamma_input: gamma_param})
                if _disc_acgan_acc < STOP_ACC_CLASS:
                    for clsitr in range(1):
                        _data,_labels = next(gen)
                        _ = session.run([class_train_op], feed_dict={all_real_data_int: datagen.flow(_data.reshape(-1, 3, 32, 32), batch_size=BATCH_SIZE, shuffle=False)[0].reshape(BATCH_SIZE, OUTPUT_DIM), all_real_labels:_labels, _iteration:iteration, gamma_input: gamma_param})
                    #_ = session.run([class_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration, gamma_input: gamma_param})
                gamma_param = min(2., max(0.0, gamma_param + 0.001*(_disc_acgan_fake - 1.0*_disc_acgan)))
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan_real', _disc_acgan)
            lib.plot.plot('acgan_fake', _disc_acgan_fake)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
            lib.plot.plot('gamma', gamma_param)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            dev_disc_acgan = []
            dev_disc_acgan_acc = []
            for images,_labels in dev_gen():
                _dev_disc_cost, _dev_disc_acgan, _dev_disc_acgan_acc = session.run([disc_cost, disc_acgan, disc_acgan_acc], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
                dev_disc_acgan.append(_dev_disc_acgan)
                dev_disc_acgan_acc.append(_dev_disc_acgan_acc)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))
            lib.plot.plot('dev_disc_acgan', np.mean(dev_disc_acgan))
            lib.plot.plot('dev_disc_acgan_acc', np.mean(dev_disc_acgan_acc))

            generate_image(iteration, _data)

        if (iteration < 20) or (iteration % 100 == 0):
        #if iteration % 1000 == 999:
            lib.plot.flush()

        lib.plot.tick()
