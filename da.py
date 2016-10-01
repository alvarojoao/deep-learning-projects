import tensorflow as tf
import numpy as np


class DenoisingAutoencoder(object):
    def __init__(self, n_input, n_hidden, layer_names,learning_rate=0.01, momentum=0.5,
        keep_prob=1.0, transfer_function_enc=tf.nn.sigmoid,transfer_function_dec=tf.nn.sigmoid,corr_type='masking',opt='gradient_descent',
        xavier_init=1,loss_func='mean_squared',corr_frac=0,regtype='none', l2reg=5e-4):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_enc = transfer_function_enc
        self.transfer_dec = transfer_function_dec

        self.layer_names = layer_names
        self.keep_prob_value = keep_prob
        self.opt = opt
        self.loss_func  = loss_func
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.xavier_init = xavier_init
        self.corr_frac = corr_frac
        self.corr_type= corr_type
        assert 0. <= self.corr_frac <= 1.

        # placeholders
        # network_weights = self._initialize_weights()
        # self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='x')
        self.x_corr = tf.placeholder(tf.float32, [None, self.n_input], name='x-corr-input')
        self.w = tf.Variable(self.xavier_init_func(self.n_input , self.n_hidden, self.xavier_init), name=self.layer_names[0])
        
        #tf.Variable(
        #        tf.truncated_normal(
        #            shape=[n_features, self.n_components], stddev=0.1),
        #        name='enc-w')
        
        self.vb = tf.Variable(tf.zeros([self.n_input ]), name=self.layer_names[1])
        self.hb = tf.Variable(tf.zeros([self.n_hidden]), name=self.layer_names[2])

        self.o_w = np.random.normal(0.0, 0.01, [self.n_input, self.n_hidden])
        self.o_vb = np.zeros([self.n_input], np.float32)
        self.o_hb = np.zeros([self.n_hidden], np.float32)

        self.keep_prob = tf.placeholder(tf.float32)
        self.weights = {}
        self.weights['w'] = self.w
        self.weights['vb'] = self.vb
        self.weights['hb'] = self.hb

        # variables
        self.encode = self.transfer_enc(tf.matmul(self.x_corr, self.w) + self.hb)
        self.encode = tf.nn.dropout(self.encode,self.keep_prob)
        self.decode = self.transfer_dec(tf.matmul(self.encode, tf.transpose(self.w)) + self.vb)
        self.decode = tf.nn.dropout(self.decode,self.keep_prob)

        if self.loss_func == 'cross_entropy':
            self.cost = - tf.reduce_sum(self.x * tf.log(self.decode))
            _ = tf.scalar_summary("cross_entropy", self.cost)
        elif self.loss_func == 'mean_squared':
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.decode)))
            _ = tf.scalar_summary("mean_squared", self.cost)
        else:
            self.cost = None

        if self.opt == 'gradient_descent':
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        elif self.opt == 'ada_grad':
            self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)
        elif self.opt == 'momentum':
            self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)
        else:
            self.train_step = None
        # cost
        # self.err_sum = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.v_sample)))
        # self.err_sum = tf.reduce_mean(tf.square(self.x - self.v_sample))

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def reconstruct(self,batch):
        return self.sess.run(self.decode,feed_dict={self.x_corr:batch,
                                                self.keep_prob:1})
    def transform(self, batch_x):
        tr_feed = {self.x_corr: batch_x, self.keep_prob:1}
        return self.sess.run(self.encode,tr_feed)


    def partial_fit(self, batch_x):
        # 1. always use small ?mini-batches? of 10 to 100 cases.
        #    For big data with lot of classes use mini-batches of size about 10.
        corruption_ratio = np.round(self.corr_frac * batch_x.shape[1]).astype(np.int)

        x_corrupted = self._corrupt_input(batch_x, corruption_ratio)
        shuff = zip(batch_x, x_corrupted)
        tr_feed = {self.x: batch_x, self.x_corr: x_corrupted,self.keep_prob:1}

        self.sess.run(self.train_step, feed_dict=tr_feed)

        return self.sess.run(self.cost, feed_dict=tr_feed)

        

    # def _initialize_weights(self):
    #     # These weights are only for storing and loading model for tensorflow Saver.
    #     all_weights = dict()
    #     all_weights['w'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden], stddev=0.01, dtype=tf.float32),
    #                                    name=self.layer_names[0])
    #     all_weights['vb'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name=self.layer_names[1])
    #     all_weights['hb'] = tf.Variable(tf.random_uniform([self.n_hidden], dtype=tf.float32), name=self.layer_names[2])
    #     return all_weights

    def _corrupt_input(self, data, v):

        """ Corrupt a fraction 'v' of 'data' according to the
        noise method of this autoencoder.
        :return: corrupted data
        """

        if self.corr_type == 'masking':
            x_corrupted = self.masking_noise(data, v)

        elif self.corr_type == 'salt_and_pepper':
            x_corrupted = self.salt_and_pepper_noise(data, v)

        elif self.corr_type == 'none':
            x_corrupted = data

        else:
            x_corrupted =  np.copy(data)

        return x_corrupted

    def xavier_init_func(self,fan_in, fan_out, const=1):
        """ Xavier initialization of network weights.
        https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        :param fan_in: fan in of the network (n_features)
        :param fan_out: fan out of the network (n_components)
        :param const: multiplicative constant
        """
        low = -const * np.sqrt(6.0 / (fan_in + fan_out))
        high = const * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)

    def masking_noise(self,X, v):
        """ Apply masking noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is forced to zero.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort
        :return: transformed data
        """
        X_noise = X.copy()

        n_samples = X.shape[0]
        n_features = X.shape[1]

        for i in range(n_samples):
            mask = np.random.randint(0, n_features, v)

            for m in mask:
                X_noise[i][m] = 0.

        return X_noise


    def salt_and_pepper_noise(self,X, v):
        """ Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
        If minimum or maximum are not given, the min (max) value in X is taken.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort
        :return: transformed data
        """
        X_noise = X.copy()
        n_features = X.shape[1]

        mn = X.min()
        mx = X.max()

        for i, sample in enumerate(X):
            mask = np.random.randint(0, n_features, v)

            for m in mask:

                if np.random.random() < 0.5:
                    X_noise[i][m] = mn
                else:
                    X_noise[i][m] = mx

        return X_noise
    
        
    def gen_image(self,img, width, height, outfile, img_type='grey'):
        assert len(img) == width * height or len(img) == width * height * 3

        if img_type == 'grey':
            misc.imsave(outfile, img.reshape(width, height))

        elif img_type == 'color':
            misc.imsave(outfile, img.reshape(3, width, height))


    def return_filters(self):
        return self.sess.run(self.w)
        
    def compute_cost(self, batch_x):
        corruption_ratio = np.round(self.corr_frac * batch_x.shape[1]).astype(np.int)
        x_corrupted = self._corrupt_input(batch_x, corruption_ratio)
        tr_feed = {self.x: batch_x, self.x_corr: x_corrupted,self.keep_prob:1}

        return self.sess.run(self.cost, feed_dict=tr_feed)

    

    def restore_weights(self, path):
        saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
                                self.layer_names[1]: self.weights['vb'],
                                self.layer_names[2]: self.weights['hb']})

        saver.restore(self.sess, path)

        self.o_w = self.weights['w'].eval(self.sess)
        self.o_vb = self.weights['vb'].eval(self.sess)
        self.o_hb = self.weights['hb'].eval(self.sess)

    def save_weights(self, path):
        self.o_w = self.weights['w'].eval(self.sess)
        self.o_vb = self.weights['vb'].eval(self.sess)
        self.o_hb = self.weights['hb'].eval(self.sess)

        self.sess.run(self.weights['w'].assign(self.o_w))
        self.sess.run(self.weights['vb'].assign(self.o_vb))
        self.sess.run(self.weights['hb'].assign(self.o_hb))
        saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
                                self.layer_names[1]: self.weights['vb'],
                                self.layer_names[2]: self.weights['hb']})
        save_path = saver.save(self.sess, path)

    def return_weights(self):
        return self.weights

    def return_hidden_weight_as_np(self):
        return self.weights['w'].eval(self.sess)
