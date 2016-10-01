import tensorflow as tf
import sys
sys.path.append('/home/ajss/notebooks/deep-learning-projects/')

from utilsnn import xavier_init


class DeepBeliefNN(object):
    def __init__(self, input_size,n_classes, layer_sizes, layer_names, finetune_learning_rate=0.001, momentum=0.5,keep_prob=1, 
                 transfer_function=tf.nn.sigmoid,l2reg=5e-4,regtype='none',loss_func='softmax_cross_entropy',opt='gradient_descent'):

        self.layer_names  = layer_names
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_value = keep_prob
        self.l2reg = l2reg
        self.regtype = regtype
        self.loss_func = loss_func
        self.opt = opt
        self.momentum = momentum
        
        # Build the encoding layers
        self.x = tf.placeholder("float", [None, input_size])
        self.y = tf.placeholder("float", [None, n_classes])
        self.finetune_learning_rate = finetune_learning_rate
        next_layer_input = self.x

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        for i in range(len(layer_sizes)):
            dim = layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using xavier initialization
            W = tf.Variable(xavier_init(input_dim, dim, transfer_function), name=layer_names[i][0])

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]), name=layer_names[i][1])

            # We are going to use tied-weights so store the W matrix for later reference.
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)
            output = tf.nn.dropout(output, self.keep_prob)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input
        self.last_W = tf.Variable(
            tf.truncated_normal(
                [self.encoded_x.get_shape()[1].value, n_classes], stddev=0.1),
            name='sm-weigths')
        self.last_b = tf.Variable(tf.constant(
                0.1, shape=[n_classes]), name='sm-biases')
        self.last_out = tf.add(tf.matmul(self.encoded_x, self.last_W),self.last_b)
        #self.layer_nodes.append(last_out)
        #self.last_out = last_out

        # build the reconstruction layers by reversing the reductions

        # compute cost
        vars = []
        vars.extend(self.encoding_matrices)
        vars.extend(self.encoding_biases)
        regterm = self.compute_regularization(vars)
        
        if self.loss_func == 'cross_entropy':
            clip_inf = tf.clip_by_value(self.last_out, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(
                    1 - self.last_out, 1e-10, float('inf'))
            cost = - tf.reduce_mean(
                    self.y  * tf.log(clip_inf+ 1e-50) +
                    (1 - self.y ) * tf.log(clip_sup+ 1e-50))

        elif self.loss_func == 'softmax_cross_entropy':
            #softmax = tf.add(tf.nn.softmax(self.last_out),1e-50)
            #sparse_
            cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.last_out,self.y))
        else:
            #mean_squared
            cost = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.last_out)))
        
        self.cost = (cost + regterm) if regterm is not None else (cost)
        
        
        if self.opt == 'gradient_descent':
            self.optimizer = tf.train.GradientDescentOptimizer(self.finetune_learning_rate).minimize(self.cost)
        elif self.opt == 'ada_grad':
            self.optimizer = tf.train.AdagradOptimizer(self.finetune_learning_rate).minimize(self.cost)
        elif self.opt == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.finetune_learning_rate, self.momentum).minimize(self.cost)
        elif self.opt == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.finetune_learning_rate).minimize(self.cost)
    
        # initalize variables
        self.model_predictions = tf.argmax(self.last_out, 1)
        correct_prediction = tf.equal(self.model_predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        _ = tf.scalar_summary('accuracy', self.accuracy)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

        
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
        
    def load_rbm_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)
        

    def print_weights(self):
        print('Matrices')
        for i in range(len(self.encoding_matrices)):
            print('Matrice',i)
            print(self.encoding_matrices[i].eval(self.sess).shape)
            print(self.encoding_matrices[i].eval(self.sess))
        print('Last Layer')
        print(self.last_W.eval(self.sess).shape)
        print(self.last_W.eval(self.sess))

            

    def load_weights(self, path):
        dict_w = self.get_dict_layer_names() 
        saver = tf.train.Saver(dict_w)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        save_path = saver.save(self.sess, path)

    def get_dict_layer_names(self):
        dict_w = {}
        for i in range(len(self.layer_names)):
            dict_w[self.layer_names[i][0]] = self.encoding_matrices[i]
            dict_w[self.layer_names[i][1]] = self.encoding_biases[i]
        dict_w['lastout-w'] = self.last_W
        dict_w['lastout-b'] = self.last_b
        return dict_w
    
    def predict(self,input_test):
        return self.model_predictions.eval({self.x: input_test,self.keep_prob: 1},session=self.sess)
    
    def predictProb(self,input_test):
        return self.last_out.eval({self.x: input_test,self.keep_prob: 1},session=self.sess)
    
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
            
    def partial_fit(self, X,y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,self.y:y,self.keep_prob:self.keep_prob_value})
        return cost
