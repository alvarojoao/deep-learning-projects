import tensorflow as tf
import sys
sys.path.append('/home/ajss/notebooks/deep-learning-projects/')

from utilsnn import xavier_init


class AutoEncoder(object):
    def __init__(self, input_size, layer_sizes, layer_names, tied_weights=False,keep_prob=1, optimizer=tf.train.AdamOptimizer(),
                 transfer_function_enc=tf.nn.relu,transfer_function_dec=tf.nn.sigmoid,l2reg=5e-4,regtype='none',loss_func='mean_squared'):

        self.layer_names  = layer_names
        self.tied_weights = tied_weights
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_value = keep_prob
        self.l2reg = l2reg
        self.regtype = regtype
        self.loss_func = loss_func
        # Build the encoding layers
        self.x = tf.placeholder("float", [None, input_size])
        next_layer_input = self.x

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        for i in range(len(layer_sizes)):
            dim = layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using xavier initialization
            W = tf.Variable(xavier_init(input_dim, dim, transfer_function_enc), name=layer_names[i][0])

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]), name=layer_names[i][1])

            # We are going to use tied-weights so store the W matrix for later reference.
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)

            output = transfer_function_enc(tf.matmul(next_layer_input, W) + b)
            output = tf.nn.dropout(output, self.keep_prob)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        self.encoding_matrices.reverse()

        self.decoding_matrices = []
        self.decoding_biases = []

        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])]):
            W = None
            # if we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            if tied_weights:
                W = tf.transpose(self.encoding_matrices[i])
            else:
                #W = tf.Variable(tf.transpose(self.encoding_matrices[i].initialized_value()))
                W = tf.Variable(xavier_init(self.encoding_matrices[i].get_shape()[1].value,self.encoding_matrices[i].get_shape()[0].value, transfer_function_dec))
            b = tf.Variable(tf.zeros([dim]))
            self.decoding_matrices.append(W)
            self.decoding_biases.append(b)

            output = transfer_function_dec(tf.matmul(next_layer_input, W) + b)
            output = tf.nn.dropout(output, self.keep_prob)
            next_layer_input = output

        # i need to reverse the encoding matrices back for loading weights
        self.encoding_matrices.reverse()
        self.decoding_matrices.reverse()

        # the fully encoded and reconstructed value of x is here:
        self.reconstructed_x = next_layer_input

        # compute cost
        vars = []
        vars.extend(self.encoding_matrices)
        vars.extend(self.encoding_biases)
        regterm = self.compute_regularization(vars)
        
        if self.loss_func == 'cross_entropy':
            clip_inf = tf.clip_by_value(self.reconstructed_x, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(
                    1 - self.reconstructed_x, 1e-10, float('inf'))
            cost = - tf.reduce_mean(
                    self.x  * tf.log(clip_inf) +
                    (1 - self.x ) * tf.log(clip_sup))

        elif self.loss_func == 'softmax_cross_entropy':
            softmax = tf.nn.softmax(self.reconstructed_x)
            cost = - tf.reduce_mean(
                    self.x  * tf.log(softmax) +
                    (1 - self.x ) * tf.log(1 - softmax))

        else:
            #mean_squared
            cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        
        self.cost = (cost + regterm) if regterm is not None else (cost)
        #_ = tf.scalar_summary(self.loss_func, self.cost)
        self.optimizer = optimizer.minimize(self.cost)

        # initalize variables
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def transform(self, X):
        return self.sess.run(self.encoded_x, {self.x: X,self.keep_prob:1})

    def reconstruct(self, X):
        return self.sess.run(self.reconstructed_x, feed_dict={self.x: X,self.keep_prob:1})

    def load_rbm_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)

        if not self.tied_weights:
            self.sess.run(self.decoding_matrices[layer].assign(tf.transpose(self.encoding_matrices[layer])))

    def print_weights(self):
        print('Matrices')
        for i in range(len(self.encoding_matrices)):
            print('Matrice',i)
            print(self.encoding_matrices[i].eval(self.sess).shape)
            print(self.encoding_matrices[i].eval(self.sess))
            if not self.tied_weights:
                print(self.decoding_matrices[i].eval(self.sess).shape)
                print(self.decoding_matrices[i].eval(self.sess))

    def load_weights(self, path):
        dict_w = self.get_dict_layer_names() 
        saver = tf.train.Saver(dict_w)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        save_path = saver.save(self.sess, path)
        
    def compute_regularization(self, vars):
        """Compute the regularization tensor.
        :param vars: list of model variables
        :return:
        """
        if self.regtype != 'none':

            regularizers = tf.constant(0.0)

            for v in vars:
                if self.regtype == 'l2':
                    regularizers = tf.add(regularizers, tf.nn.l2_loss(v))
                elif self.regtype == 'l1':
                    regularizers = tf.add(
                        regularizers, tf.reduce_sum(tf.abs(v)))

            return tf.mul(self.l2reg, regularizers)
        else:
            return None
        
    def get_dict_layer_names(self):
        dict_w = {}
        for i in range(len(self.layer_names)):
            dict_w[self.layer_names[i][0]] = self.encoding_matrices[i]
            dict_w[self.layer_names[i][1]] = self.encoding_biases[i]
            if not self.tied_weights:
                dict_w[self.layer_names[i][0]+'d'] = self.decoding_matrices[i]
                dict_w[self.layer_names[i][1]+'d'] = self.decoding_biases[i]
        return dict_w

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,self.keep_prob:self.keep_prob_value})
        return cost
