import os, sys
import numpy as np
import theano
import theano.tensor as T
import h5py
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               # The name parameter is solely for printing purposes
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               # Theano allows for broadcasting, similar to numpy.
                               # However, you need to explicitly denote which axes can be broadcasted.
                               # By setting broadcastable=(False, True), we are denoting that b
                               # can be broadcast (copied) along its second dimension in order to be
                               # added to another variable.  For more information, see
                               # http://deeplearning.net/software/theano/library/tensor/basic.html
                               broadcastable=(False, True))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]

    def output(self, x):
        '''
        Compute this layer's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        #return (lin_output if self.activation is None else self.activation(lin_output))

        if self.activation is T.nnet.softmax:
            # Softmax is applied row wise
                lin_output = self.activation(lin_output.T).T
        elif not self.activation is None :
                lin_output = self.activation(lin_output)
    
        return lin_output
                

class MLP(object):
    def __init__(self, W_init, b_init, activations, activation_str=None):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers
        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)

        # Initialize lists of layers
        self.layers = []
        self.activation_str = activation_str
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # self.norms = []
        self.in_norms = []
        self.out_norms = []
        self.trans_mat = []

    def output(self, x):
        '''
        Compute the MLP's output given an input
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def output_layer_id(self, x, layer_id):
        '''
        Compute the MLP's output given an input
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers[:layer_id]:
            x = layer.output(x)
        return x

    def XEnt(self,x,y):
        # return T.log(self.output(x)[y])
        return T.nnet.crossentropy_categorical_1hot(self.output(x).T,y)
    def classification_accuracy(self,x,y):
        return T.eq(T.argmax(self.output(x),axis=0) ,y)
    def sum_squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output
        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)

    def mean_squared_error(self, x, y):
        return T.mean((self.output(x) - y)**2)

    def L1_norm(self):
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        L1 = 0
        for layer in self.layers:
            L1 += T.sum(abs(layer.W))
        return L1

    def sqr_L2_norm(self):
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        L2 = 0
        for layer in self.layers:
            L2 += T.sum(layer.W ** 2)
        return L2

    def layer_mean(self):
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        means = []
        for layer in self.layers:
            means.append(T.mean(layer.W))
        return means

    def layer_var(self):
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        var = []
        for layer in self.layers:
            var.append(T.var(layer.W))
        return var 

    def save_model(self, save_dir, filename):
        '''
        Save the parameters of the model
        '''
        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        f = h5py.File(os.path.join(save_dir, filename), 'w')

        for count in range(len(self.layers)):
            dset = f.create_dataset('params/' + str(count) + '/W', data=self.layers[count].params[0].get_value(borrow=True), dtype=np.float32)
            dset = f.create_dataset('params/' + str(count) + '/b', data=self.layers[count].params[1].get_value(borrow=True), dtype=np.float32)

        dset = f.create_dataset('norms/in', data=self.in_norms, dtype=np.float32)
        dset = f.create_dataset('norms/out', data=self.out_norms, dtype=np.float32)
        dset = f.create_dataset('trans_mat', data=self.trans_mat, dtype=np.float32)

        f.close()

    def load_model(self, load_dir, filename):
        '''
        Load the parameters
        '''
        print '... loading model'

        f = h5py.File(os.path.join(load_dir, filename), 'r')

        for count in range(len(self.layers)):
            self.layers[count].params[0].set_value(f['params/' + str(count) + '/W'][...], borrow=True)
            self.layers[count].params[1].set_value(f['params/' + str(count) + '/b'][...], borrow=True)

        # self.norms = f['norms'][...]
        self.in_norms = f['norms/in'][...]
        self.out_norms = f['norms/out'][...]
        self.trans_mat = f['trans_mat'][...]

        f.close()

    def load_model_except_last(self, load_dir, filename):
        '''
        Load the parameters except the parameters of last new layers
        '''
        print '... loading model'

        f = h5py.File(os.path.join(load_dir, filename), 'r')

        for count in range(len(self.layers) - 2):
            self.layers[count].params[0].set_value(f['params/' + str(count) + '/W'][...], borrow=True)
            self.layers[count].params[1].set_value(f['params/' + str(count) + '/b'][...], borrow=True)

        # self.norms = f['norms'][...]
        self.in_norms = f['norms/in'][...]
        self.out_norms = f['norms/out'][...]
        self.trans_mat = f['trans_mat'][...]

        f.close()


class MLP_wDO(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers
        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)

        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # self.norms = []
        self.in_norms = []
        self.out_norms = []
        self.trans_mat = []

    def output(self, x, dropout_rate=0., use_dropout=False):
        '''
        Compute the MLP's output given an input
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output of hidden layers
        for layer in self.layers[:-1]:
            x = layer.output(x)
            if dropout_rate != 0.:
                if use_dropout:     # for training
                    x *= T.cast(MRG_RandomStreams(use_cuda=True).binomial(n=1, p=(1.-dropout_rate), size=x.shape), 'int8')
                else:               # for validation and testing
                    x *= T.cast(1.-dropout_rate, theano.config.floatX)

        # Compute output of output (last) layer
        x = self.layers[-1].output(x)

        return x

    def sum_squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output
        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)

    def mean_squared_error(self, x, y):
        return T.mean((self.output(x) - y)**2)

    def L1_norm(self):
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        L1 = 0
        for layer in self.layers:
            L1 += T.sum(abs(layer.W))
        return L1

    def sqr_L2_norm(self):
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        L2 = 0
        for layer in self.layers:
            L2 += T.sum(layer.W ** 2)
        return L2

    def save_model(self, save_dir, filename):
        '''
        Save the parameters of the model
        '''
        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        f = h5py.File(os.path.join(save_dir, filename), 'w')

        for count in range(len(self.layers)):
            dset = f.create_dataset('params/' + str(count) + '/W', data=self.layers[count].params[0].get_value(borrow=True), dtype=np.float32)
            dset = f.create_dataset('params/' + str(count) + '/b', data=self.layers[count].params[1].get_value(borrow=True), dtype=np.float32)

        dset = f.create_dataset('norms/in', data=self.in_norms, dtype=np.float32)
        dset = f.create_dataset('norms/out', data=self.out_norms, dtype=np.float32)
        dset = f.create_dataset('trans_mat', data=self.trans_mat, dtype=np.float32)

        f.close()

    def load_model(self, load_dir, filename):
        '''
        Load the parameters
        '''
        print '... loading model'

        f = h5py.File(os.path.join(load_dir, filename), 'r')

        for count in range(len(self.layers)):
            self.layers[count].params[0].set_value(f['params/' + str(count) + '/W'][...], borrow=True)
            self.layers[count].params[1].set_value(f['params/' + str(count) + '/b'][...], borrow=True)

        # self.norms = f['norms'][...]
        self.in_norms = f['norms/in'][...]
        self.out_norms = f['norms/out'][...]
        self.trans_mat = f['trans_mat'][...]

        f.close()

    def load_model_except_last(self, load_dir, filename):
        '''
        Load the parameters except the parameters of last new layers
        '''
        print '... loading model'

        f = h5py.File(os.path.join(load_dir, filename), 'r')

        for count in range(len(self.layers) - 2):
            self.layers[count].params[0].set_value(f['params/' + str(count) + '/W'][...], borrow=True)
            self.layers[count].params[1].set_value(f['params/' + str(count) + '/b'][...], borrow=True)

        # self.norms = f['norms'][...]
        self.in_norms = f['norms/in'][...]
        self.out_norms = f['norms/out'][...]
        self.trans_mat = f['trans_mat'][...]

        f.close()


def gradient_updates_momentum(cost, params, param_updates, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for c in xrange(len(params)):
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        # param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        #print param_update.get_value()
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((params[c], params[c] - learning_rate*param_updates[c]))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_updates[c], momentum*param_updates[c] + (1. - momentum)*T.grad(cost, params[c])))
    return updates


''' Classical momentum '''
def SGD_CM_updates(cost, params, param_updates, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for c in xrange(len(params)):
        param_updates_plus1 = momentum*param_updates[c] - learning_rate*T.grad(cost, params[c])
        updates.append((param_updates[c], param_updates_plus1))
        updates.append((params[c], params[c] + param_updates_plus1))
    return updates


''' Nesterov's Accelerated Gradient '''
def NAG_updates(cost, params, param_updates, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for c in xrange(len(params)):
        # see https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
        param_updates_plus1 = momentum*param_updates[c] - learning_rate*T.grad(cost, params[c])
        updates.append((param_updates[c], param_updates_plus1))
        updates.append((params[c], params[c] + momentum*param_updates_plus1 - learning_rate*T.grad(cost, params[c])))
    return updates


''' ADADELTA '''
def ADADELTA_updates(cost, params, grad_accums, updt_accums, rho=0.95, eps=1e-6):
    updates = []
    for c in xrange(len(params)):
        local_grad = T.grad(cost, params[c])
        local_grad_accums = rho * grad_accums[c] + (1 - rho) * T.sqr(local_grad)
        local_param_update = -1 * (T.sqrt(updt_accums[c] + eps) / T.sqrt(local_grad_accums + eps) ) * local_grad
        local_updt_accums =rho * updt_accums[c] + (1 - rho) * T.sqr(local_param_update)
        updates.append((params[c], params[c] + local_param_update))
        updates.append((grad_accums[c], local_grad_accums))
        updates.append((updt_accums[c], local_updt_accums))
    return updates


def ReLU(x):
    return T.switch(x < 0, 0, x)


def sReLU(x):
    return T.log(1+T.exp(x))
