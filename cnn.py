import tensorflow as tf
import numpy as np


class ConvolutionalNetwork(object):
    """Constructor.

    :param layers: ex: conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-1024,softmax string used to build the model.
        This string is a comma-separate specification of the layers.
        Supported values: 
            conv2d-FX-FY-Z-S: 2d convolution with Z feature maps as output
                and FX x FY filters. S is the strides size
            maxpool-X: max pooling on the previous layer. X is the size of
                the max pooling
            full-X: fully connected layer with X units
            softmax: softmax layer
        For example:
            conv2d-5-5-32,maxpool-2,conv2d-5-5-64,maxpool-2,full-128,full-128,softmax

    :param original_shape: ex: 28,28,1 original shape of the images in the dataset
    :param dropout: Dropout parameter
    :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
    """
    def __init__(self, layers,  n_features, n_classes,original_shape,
        learning_rate=0.01, momentum=0.5,
        keep_prob=1.0, opt='gradient_descent', 
        loss_func='softmax_cross_entropy'):

        self.layers = layers
        self.original_shape = original_shape

        self.keep_prob_value = keep_prob
        self.opt = opt
        self.loss_func  = loss_func
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.encode = None
        # placeholders
        self.x = tf.placeholder(
            tf.float32, [None, n_features], name='x-input')
        self.y = tf.placeholder(
            tf.float32, [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep-probs')

        self._create_layers(n_classes)

        if self.loss_func == 'cross_entropy':
            clip_inf = tf.clip_by_value(self.last_out, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(
                1 - self.last_out, 1e-10, float('inf'))
            self.cost = - tf.reduce_mean(
                self.y * tf.log(clip_inf) +
                (1 - self.y) * tf.log(clip_sup))
        elif self.loss_func == 'softmax_cross_entropy':
            # softmax = tf.nn.softmax(self.last_out)
            # cost = - tf.reduce_mean(
            #     self.y * tf.log(softmax) +
            #     (1 - self.y) * tf.log(1 - softmax))
            self.cost = tf.contrib.losses.softmax_cross_entropy(self.last_out,self.y)
        elif self.loss_func == 'mean_squared':
            self.cost = tf.sqrt(tf.reduce_mean(
                tf.square(self.y - self.last_out)))

        if self.opt == 'gradient_descent':
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        elif self.opt == 'ada_grad':
            self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)
        elif self.opt == 'momentum':
            self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)
        elif self.opt == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.cost)
        else:
            self.train_step = None
        
        self.model_predictions = tf.argmax(self.last_out, 1)
        correct_prediction = tf.equal(self.model_predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, "float"))
        _ = tf.scalar_summary('accuracy', self.accuracy)


        init = tf.initialize_all_variables()

        self.tf_saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

    
    def predict(self, batch_x):
        tr_feed = {self.x: batch_x, self.keep_prob:1}
        return self.sess.run(self.model_predictions,tr_feed)

    def predictProb(self,batch_x):
        tr_feed = {self.x: batch_x, self.keep_prob:1}
        return self.sess.run(self.last_out,tr_feed)

    def accuracy(self,batch_x,batch_y):
        tr_feed = {self.x: batch_x, self.y: batch_y, self.keep_prob:1}
        return self.sess.run(self.accuracy,tr_feed)

    def transform(self, batch_x):
        tr_feed = {self.x: batch_x, self.keep_prob:1}
        return self.sess.run(self.encode,tr_feed)
    
    def saveWeights(self,model_path):
        self.tf_saver.save(self.sess, model_path)
        return None

    def restoreWeights(self,model_path):
        self.tf_merged_summaries = tf.merge_all_summaries()
        self.tf_saver.restore(self.tf_session, self.model_path)
        return None
    
    def load_weights(self, path):
        dict_w = self.get_dict_layer_names() 
        saver = tf.train.Saver(dict_w)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        save_path = saver.save(self.sess, path)
    def get_dict_layer_names(self):
        a = self.layers.split(',')
        self.W_vars = []
        self.B_vars = []
        
    def _create_layers(self, n_classes):
        """Create the layers of the model from self.layers.

        :param n_classes: number of classes
        :return: self
        """
        next_layer_feed = tf.reshape(self.x,
                                     [-1, self.original_shape[0],
                                      self.original_shape[1],
                                      self.original_shape[2]])
        prev_output_dim = self.original_shape[2]
        # this flags indicates whether we are building the first dense layer
        first_full = True

        self.W_vars = []
        self.B_vars = []

        for i, l in enumerate(self.layers.split(',')):

            node = l.split('-')
            node_type = node[0]

            if node_type == 'conv2d':

                # ################### #
                # Convolutional Layer #
                # ################### #

                # fx, fy = shape of the convolutional filter
                # feature_maps = number of output dimensions
                fx, fy, feature_maps, stride = int(node[1]),\
                     int(node[2]), int(node[3]), int(node[4])

                print('Building Convolutional layer with %d input channels\
                      and %d %dx%d filters with stride %d' %
                      (prev_output_dim, feature_maps, fx, fy, stride))

                # Create weights and biases
                W_conv = self.weight_variable(
                    [fx, fy, prev_output_dim, feature_maps])
                b_conv = self.bias_variable([feature_maps])
                self.W_vars.append(W_conv)
                self.B_vars.append(b_conv)

                # Convolution and Activation function
                h_conv = tf.nn.relu(
                    self.conv2d(next_layer_feed, W_conv, stride) + b_conv)

                # keep track of the number of output dims of the previous layer
                prev_output_dim = feature_maps
                # output node of the last layer
                next_layer_feed = h_conv

            elif node_type == 'maxpool':

                # ################# #
                # Max Pooling Layer #
                # ################# #

                ksize = int(node[1])

                print('Building Max Pooling layer with size %d' % ksize)

                next_layer_feed = self.max_pool(next_layer_feed, ksize)

            elif node_type == 'full':

                # ####################### #
                # Densely Connected Layer #
                # ####################### #

                if first_full:  # first fully connected layer

                    dim = int(node[1])
                    shp = next_layer_feed.get_shape()
                    tmpx = shp[1].value
                    tmpy = shp[2].value
                    fanin = tmpx * tmpy * prev_output_dim

                    print('Building fully connected layer with %d in units\
                          and %d out units' % (fanin, dim))

                    W_fc = self.weight_variable([fanin, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_pool_flat = tf.reshape(next_layer_feed, [-1, fanin])
                    h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

                    first_full = False

                else:  # not first fully connected layer

                    dim = int(node[1])
                    W_fc = self.weight_variable([prev_output_dim, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_fc = tf.nn.relu(tf.matmul(next_layer_feed, W_fc) + b_fc)
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

            elif node_type == 'softmax':

                # ############# #
                # Softmax Layer #
                # ############# #

                print('Building softmax layer with %d in units and\
                      %d out units' % (prev_output_dim, n_classes))

                W_sm = self.weight_variable([prev_output_dim, n_classes])
                b_sm = self.bias_variable([n_classes])
                self.W_vars.append(W_sm)
                self.B_vars.append(b_sm)
                self.encode = next_layer_feed
                self.last_out = tf.matmul(next_layer_feed, W_sm) + b_sm


    def partial_fit(self, batch_x,batch_y):
        # 1. always use small ?mini-batches? of 10 to 100 cases.
        #    For big data with lot of classes use mini-batches of size about 10.

        tr_feed = {self.x: batch_x, self.y: batch_y,self.keep_prob:1}

        self.sess.run(self.train_step, feed_dict=tr_feed)

        return self.sess.run(self.cost, feed_dict=tr_feed)

    @staticmethod
    def weight_variable(shape):
        """Create a weight variable."""
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Create a bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        """2D Convolution operation."""
        return tf.nn.conv2d(
            x, W, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def max_pool(x, dim):
        """Max pooling operation."""
        return tf.nn.max_pool(
            x, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1],
            padding='SAME')

 
    def compute_cost(self, batch_x,batch_y):
        tr_feed = {self.x: batch_x, self.y: batch_y,self.keep_prob:1}

        return self.sess.run(self.cost, feed_dict=tr_feed)

    
 

 
