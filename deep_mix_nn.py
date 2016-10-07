import tensorflow as tf
import sys
sys.path.append('/home/ajss/notebooks/deep-learning-projects/')

from utilsnn import xavier_init
import numpy as np

class DeepMixNN(object):
    def __init__(self, input_size,n_classes, layer_sizes, layer_names, tied_weights=False,keep_prob=1, momentum=0.5,
                 opt_unsup='gradient_descent',opt_sup='ada_grad',finetune_learning_rate=0.3,
                 transfer_function_enc=tf.nn.sigmoid,transfer_function_dec=tf.nn.sigmoid,l2reg=5e-4,regtype='none'
                 ,loss_func_target='softmax_cross_entropy',loss_func_au='mean_squared',corr_frac=0,corr_type='none',dir_='mixDbn'):

        self.layer_names  = layer_names
        self.tied_weights = tied_weights
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_value = keep_prob
        self.l2reg = l2reg
        self.regtype = regtype
        self.loss_func_au = loss_func_au
        self.loss_func_target = loss_func_target
        self.finetune_learning_rate = finetune_learning_rate
        self.opt_unsup = opt_unsup
        self.opt_sup = opt_sup
        self.momentum = momentum
        
        self.corr_frac = corr_frac
        self.corr_type= corr_type
        
        # Build the encoding layers
        self.x = tf.placeholder(tf.float32, [None, input_size],name='x-input')
        self.y = tf.placeholder(tf.float32, [None, n_classes],name='y-input')
        self.x_corr = tf.placeholder(tf.float32, [None, input_size], name='x-corr-input')

        #next_layer_input = self.x
        next_layer_input = self.x_corr

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        self.enconding_vars = []
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
            
            self.enconding_vars.append(W)
            self.enconding_vars.append(b)
            
            output = transfer_function_enc(tf.matmul(next_layer_input, W) + b)
            output = tf.nn.dropout(output, self.keep_prob)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        with tf.name_scope('Model'):
            # The fully encoded x value is now stored in the next_layer_input
            self.encoded_x = next_layer_input
        
        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        layer_names.reverse()
        self.encoding_matrices.reverse()
        
        self.decoding_matrices = []
        self.decoding_biases = []
        self.decoding_vars = []
        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])]):
            W = None
            # if we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            if tied_weights:
                W = tf.transpose(self.encoding_matrices[i])
                b = tf.Variable(tf.zeros([dim]))
            else:
                #W = tf.Variable(tf.transpose(self.encoding_matrices[i].initialized_value()))
                W = tf.Variable(xavier_init(self.encoding_matrices[i].get_shape()[1].value,
                                            self.encoding_matrices[i].get_shape()[0].value, 
                                            transfer_function_dec),name=layer_names[i][0]+'d')
                b = tf.Variable(tf.zeros([dim]),name=layer_names[i][1]+'d')
            self.decoding_matrices.append(W)
            self.decoding_biases.append(b)
            self.decoding_vars.append(W)
            self.decoding_vars.append(b)
            
            output = transfer_function_dec(tf.matmul(next_layer_input, W) + b)
            output = tf.nn.dropout(output, self.keep_prob)
            next_layer_input = output

        # i need to reverse the encoding matrices back for loading weights
        self.encoding_matrices.reverse()
        self.decoding_matrices.reverse()
    
        # the fully encoded and reconstructed value of x is here:
        self.reconstructed_x = next_layer_input
        
        # The fully encoded x value is now stored in the next_layer_input
        self.last_W = tf.Variable(
            tf.truncated_normal(
                [self.encoded_x.get_shape()[1].value, n_classes], stddev=0.1),
            name='sm-weigths')
        self.last_b = tf.Variable(tf.constant(
                0.1, shape=[n_classes]), name='sm-biases')
        with tf.name_scope('Model'):
            self.last_out = tf.add(tf.matmul(self.encoded_x, self.last_W),self.last_b)
        
        # compute cost
        vars = []
        vars.extend(self.encoding_matrices)
        vars.extend(self.encoding_biases)
        regterm = self.compute_regularization(vars)

        if self.loss_func_au == 'cross_entropy':
            clip_inf = tf.clip_by_value(self.reconstructed_x, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(
                    1 - self.reconstructed_x, 1e-10, float('inf'))
            cost_au = - tf.reduce_mean(
                    self.x  * tf.log(clip_inf) +
                    (1 - self.x ) * tf.log(clip_sup))

        elif self.loss_func_au == 'softmax_cross_entropy':
            cost_au =  tf.contrib.losses.softmax_cross_entropy(self.reconstructed_x,self.x)
        else:
            #mean_squared
            cost_au = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        
        if self.loss_func_target == 'cross_entropy':
            clip_inf = tf.clip_by_value(self.last_out, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(
                    1 - self.last_out, 1e-10, float('inf'))
            cost_target = - tf.reduce_mean(
                    self.y  * tf.log(clip_inf) +
                    (1 - self.y ) * tf.log(clip_sup))

        elif self.loss_func_target == 'softmax_cross_entropy':
            cost_target =  tf.contrib.losses.softmax_cross_entropy(self.last_out,self.y)
        else:
            #mean_squared
            cost_target = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.last_out)))
         
        # cost = cost_target# tf.add(cost_au,cost_target)
        with tf.name_scope('Loss-au'):
            self.cost_target = (cost_target + regterm) if regterm is not None else (cost_target)
        with tf.name_scope('Loss-target'):
            self.cost_au = (cost_au + regterm) if regterm is not None else (cost_au)
        #_ = tf.scalar_summary(self.loss_func, self.cost)
        
        with tf.name_scope('OPT-au'):
            if self.opt_sup == 'gradient_descent':
                all_var  = self.enconding_vars + [self.last_W,self.last_b]
                self.train_step_sup = tf.train.GradientDescentOptimizer(self.finetune_learning_rate).minimize(self.cost_target,var_list=all_var)
            elif self.opt_sup == 'ada_grad':
                all_var  = self.enconding_vars + [self.last_W,self.last_b]
                self.train_step_sup = tf.train.AdagradOptimizer(self.finetune_learning_rate).minimize(self.cost_target,var_list=all_var)
            elif self.opt_sup == 'momentum':
                all_var  = self.enconding_vars + [self.last_W,self.last_b]
                self.train_step_sup = tf.train.MomentumOptimizer(self.finetune_learning_rate, self.momentum).minimize(self.cost_target,var_list=all_var)
            elif self.opt_sup == 'adam':
                all_var  = self.enconding_vars + [self.last_W,self.last_b]
                self.train_step_sup = tf.train.AdamOptimizer(
                    self.finetune_learning_rate).minimize(self.cost_target,var_list=all_var)
            else:
                self.train_step_sup = None
        
        with tf.name_scope('OPT-target'):
            if self.opt_unsup == 'gradient_descent':
                all_var  = self.enconding_vars + self.decoding_vars
                self.train_step_unsup = tf.train.GradientDescentOptimizer(self.finetune_learning_rate).minimize(self.cost_au,var_list=all_var)
            elif self.opt_unsup == 'ada_grad':
                all_var  = self.enconding_vars + self.decoding_vars
                self.train_step_unsup = tf.train.AdagradOptimizer(self.finetune_learning_rate).minimize(self.cost_au,var_list=all_var)
            elif self.opt_unsup == 'momentum':
                all_var  = self.enconding_vars + self.decoding_vars
                self.train_step_unsup = tf.train.MomentumOptimizer(self.finetune_learning_rate, self.momentum).minimize(self.cost_au,var_list=all_var)
            elif self.opt_unsup == 'adam':
                all_var  = self.enconding_vars + self.decoding_vars
                self.train_step_unsup = tf.train.AdamOptimizer(
                    self.finetune_learning_rate).minimize(self.cost_au,var_list=all_var)
            else:
                self.train_step_unsup = None
            
        self.model_predictions = tf.argmax(self.last_out, 1)
        correct_prediction = tf.equal(self.model_predictions, tf.argmax(self.y, 1))
        with tf.name_scope('Accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        self.loss_unsup_summary = tf.scalar_summary("loss-au", self.cost_au)
        self.loss_sup_summary = tf.scalar_summary("loss-sup", self.cost_target)
        # self.merged_summary_op = tf.merge_all_summaries()
        
        self.accuracy_summary = tf.scalar_summary('accuracy', self.accuracy)
        
#         scaledImageRec_x = tf.image.convert_image_dtype(self.reconstructed_x, dtype=tf.uint8)
#         reconstructed_x_transposed = tf.transpose(scaledImageRec_x, [None, 28, 28, 1])
#         self.reconstruct_summary = tf.image_summary('reconstruct', reconstructed_x_transposed ,max_image=30) 
        
        logs_path = './tf-logs/'+dir_
        # initalize variables
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        self.summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())


    def transform(self, X):
        return self.sess.run(self.encoded_x, {self.x: X,self.x_corr: X,self.keep_prob:1})
    
    def predict(self,input_test):
        return self.model_predictions.eval({self.x: input_test,self.x_corr:input_test,self.keep_prob: 1},session=self.sess)
    
    def predictProb(self,input_test):
        return self.last_out.eval({self.x: input_test,self.x_corr:input_test,self.keep_prob: 1},session=self.sess)
    
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
    
    def compute_accuracy(self, test_set, test_labels):
        """Compute the accuracy over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_labels: Labels for the test data.
            shape(n_test_samples, n_classes)
        :return: accuracy
        """
        return self.accuracy.eval({self.x: test_set,
                                   self.y: test_labels,
                                   self.keep_prob: 1},session=self.sess)
    
    def reconstruct(self, X):
        return self.sess.run(self.reconstructed_x, feed_dict={self.x: X,self.x_corr: X,self.keep_prob:1})

    def load_rbm_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)

        if not self.tied_weights:
            self.sess.run(self.decoding_matrices[layer].assign(tf.transpose(self.encoding_matrices[layer])))
    
    def load_da_weights(self, path, layer_names, layer):
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
            dict_w['sm-weigths'] = self.last_W 
            dict_w['sm-biases'] = self.last_b
            if not self.tied_weights:
                dict_w[self.layer_names[i][0]+'d'] = self.decoding_matrices[i]
                dict_w[self.layer_names[i][1]+'d'] = self.decoding_biases[i]
        return dict_w

    def partial_fit(self, batch_x,batch_y,valid_x,valid_y,epoch=0,i=0,batch_size=1000):
        # 1. always use small ?mini-batches? of 10 to 100 cases.
        #    For big data with lot of classes use mini-batches of size about 10.

        
        corruption_ratio = np.round(self.corr_frac * batch_x.shape[1]).astype(np.int)

        x_corrupted = self._corrupt_input(batch_x, corruption_ratio)
        
        
        tr_feed = {self.x: batch_x, self.y: batch_y,self.x_corr:x_corrupted,self.keep_prob:self.keep_prob_value}
        self.sess.run(self.train_step_unsup, feed_dict=tr_feed)
        # cost_au = self.sess.run(self.cost_au, feed_dict=tr_feed)
        loss_unsup_summary = self.sess.run(self.loss_unsup_summary, feed_dict=tr_feed)
        self.summary_writer.add_summary(loss_unsup_summary, epoch * batch_size + i)
        
        # summary = self.sess.run(self.merged_summary_op, feed_dict=tr_feed)
        # self.summary_writer.add_summary(summary, epoch * batch_size + i)

        
        

        tr_feed = {self.x: batch_x, self.y: batch_y,self.x_corr:batch_x,self.keep_prob:self.keep_prob_value}
        self.sess.run(self.train_step_sup, feed_dict=tr_feed)
        # cost_target = self.sess.run(self.cost_target, feed_dict=tr_feed)
        loss_sup_summary = self.sess.run(self.loss_sup_summary, feed_dict=tr_feed)
        self.summary_writer.add_summary(loss_sup_summary, epoch * batch_size + i)
        
        
        
        tr_feed = {self.x: valid_x, self.y: valid_y,self.x_corr:valid_x,self.keep_prob:1}
        accuracy_summary = self.sess.run(self.accuracy_summary, feed_dict=tr_feed)
        self.summary_writer.add_summary(accuracy_summary, epoch * batch_size + i)
        
        # if (epoch * batch_size + i)%2000==0:
        #     tr_feed = {self.x: valid_x, self.y: valid_y,self.x_corr:valid_x,self.keep_prob:1}
        #     reconstruct_summary = self.sess.run(self.reconstruct_summary, feed_dict=tr_feed)
        #     self.summary_writer.add_summary(reconstruct_summary, epoch * batch_size + i)

    
    
    def compute_cost(self, batch_x,batch_y):
        tr_feed = {self.x: batch_x, self.y: batch_y,self.keep_prob:1}
        cost_target = self.sess.run(self.cost_target, feed_dict=tr_feed)
        cost_au = self.sess.run(self.cost_au, feed_dict=tr_feed)
        return (cost_target,cost_au)
    
