import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, name, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = lib.param(name.replace('Discriminator.','D.').replace('Generator.','G.').replace('Classifier.','C.')+".u", np.random.normal(size=(1, w_shape[-1])).astype('float32'), trainable=False)

   u_hat = tf.identity(u)
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = l2_norm(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = l2_norm(u_)

   u_final = tf.identity(u_hat)
   v_final = tf.identity(v_hat)

   u_final = tf.stop_gradient(u_final)
   v_final = tf.stop_gradient(v_final)

   sigma = tf.matmul(tf.matmul(v_final, w), tf.transpose(u_final))

   assign_u = tf.compat.v1.assign(u, u_final)
   with tf.control_dependencies([assign_u]):
       sigma = tf.identity(sigma)
       w_norm = tf.identity(w / sigma)
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm

def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization='glorot_he',
        weightnorm=None,
        gain=1.,
        s_norm=False
        ):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot' or (initialization == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')
            weight_values = sample((input_dim, output_dim))
        
        elif initialization[0] == 'uniform':
        
            weight_values = np.random.uniform(
                low=-initialization[1],
                high=initialization[1],
                size=(input_dim, output_dim)
            ).astype('float32')

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = lib.param(
            name + '.W',
            weight_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = lib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(name)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if inputs.get_shape().ndims == 2:
            if s_norm:
                result = tf.matmul(inputs, spectral_norm(weight, name))
            else:
                result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            if s_norm:
                result = tf.matmul(reshaped_inputs, spectral_norm(weight, name))
            else:
                result = tf.matmul(inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                lib.param(
                    name + '.b',
                    np.zeros((output_dim,), dtype='float32')
                )
            )

        return result