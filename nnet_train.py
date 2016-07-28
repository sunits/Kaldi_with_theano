# Author: Sunit Sivasankaran
# Instituition : Inria - Nancy

from optparse import OptionParser
import MLP
import theano
import readKaldiDNN
import sys
import h5py
import readKaldiData
import numpy as np
import theano.tensor as T
import ipdb
from utils import *
import time
import os


RDMSEED = 111214
mini_batch_size = 256
DEV_MINI_EPOCH = 100
MINI_EPOCH = 2000
DO_VALIDATION_EVERY_N_MINI_EPOCH = 1
training_percentage = 1
dev_percentage = 1
MAX_EPOCH = 15
CONTEXT = [5, 5] # Splicing context
momentum = 0.99
learning_rate_val = 0.008
delta = [0., 0.]
delta_bar = [0., 0.]
E_valid = [0., 0.]
decrement_lr = 0.7
increment_lr = 1.1
theta_lr = 1e-3
alpha_lr = 1e-1
patience_lr = 5
max_patience = patience_lr * 3

print 'Learning rate', learning_rate_val


# All possible activation units
activation_dict = {}
activation_dict['Sigmoid'] = T.nnet.sigmoid
activation_dict['Relu'] = T.nnet.relu
activation_dict['Softmax'] = T.nnet.softmax


def create_context(X):
    left_feats = [X]
    right_feats = [X]
    for index in xrange(CONTEXT[0]):
        temp = np.roll(left_feats[-1], 1 ,axis=1)
        temp[:, 0] = left_feats[-1][:, 0]
        left_feats.append(temp)
    for index in xrange(CONTEXT[1]):
        temp = np.roll(right_feats[-1], -1, axis=1)
        temp[:, -1] = right_feats[-1][:, -1]
        right_feats.append(temp)
    return np.vstack((np.vstack([ele for ele in reversed(left_feats[1:])]), \
            X, np.vstack([ele for ele in right_feats[1:]])))


# Loading the class priors
# This is usefull which decoding
def load_kaldi_priors(path, prior_cutoff, uniform_smoothing_scaler=0.05):
    assert 0 <= uniform_smoothing_scaler <= 1.0, (
        "Expected 0 <= uniform_smoothing_scaler <=1, got %f" % uniform_smoothing_scaler
    )
    numbers = np.fromregex(path, r"([\d\.e+]+)", dtype=[('num', np.float32)])
    class_counts = np.asarray(numbers['num'], dtype=theano.config.floatX)
    # compute the uniform smoothing count
    uniform_priors = np.ceil(class_counts.mean() * uniform_smoothing_scaler)
    priors = (class_counts + uniform_priors) / class_counts.sum()
#floor zeroes to something small so log() on that will be different 
# from -inf or better skip these in contribution at all i.e. set to -log(0)?
    priors[priors < prior_cutoff] = prior_cutoff
    assert np.all(priors > 0) and np.all(priors <= 1.0), (
        "Prior probabilities outside [0,1] range."
    )
    log_priors = np.log(priors)
    assert not np.any(np.isinf(log_priors)), (
        "Log-priors contain -inf elements."
    )
    return log_priors

# Defining NN given the weigts (obtained using RBM perhaps)
def define_nn_preset(W_init, b_init, activations):
    return MLP.MLP(W_init, b_init, activations)

def define_plain_nn_preset(W_init, b_init, activations):
    return MLP.MLP(W_init, b_init, activations)

def define_nn(n_input_dims, n_hidden_dims, n_hidden_layer, n_output_dims, W_init=None, b_init=None):
    layer_sizes = [n_input_dims]
    activations = []
    for i in xrange(0, n_hidden_layer):
        layer_sizes.append(n_hidden_dims)
        activations.append(T.nnet.sigmoid)

    layer_sizes.append(n_output_dims)
    activations.append(T.nnet.softmax)

    # Set initial parameter values
    act_count = 0
    if W_init is None:
        W_init = []
        b_init = []
        for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
            if activations[act_count] == T.tanh:
                W_init.append(np.random.RandomState(RDMSEED).uniform(
                    low=-np.sqrt(6. / (n_output + n_input)), 
                    high=np.sqrt(6. / (n_output + n_input)),
                    size=(n_output, n_input)))
            elif activations[act_count] == T.nnet.sigmoid:
                W_init.append(np.random.RandomState(RDMSEED).uniform(
                    low=-4*np.sqrt(6. / (n_output + n_input)), 
                    high=4*np.sqrt(6. / (n_output + n_input)),
                    size=(n_output, n_input)))
            elif activations[act_count] == T.nnet.relu:
                W_init.append(np.random.RandomState(RDMSEED).normal(0., np.sqrt(2./n_input), (n_output, n_input)))
            else:
                W_init.append(np.random.RandomState(RDMSEED).normal(0., 0.01, (n_output, n_input)))
            act_count += 1
            b_init.append(np.zeros(n_output))
    mlp = MLP.MLP(W_init, b_init, activations)
    return mlp


def main(args=None):
    usage = args[0] + " [options] <model> <feat.scp> <align_dir>  \
                        <align_kaldi_model> <adapted_model_path>"
    parser = OptionParser(usage)
    parser.add_option("--feature-transform", dest="feature_transform", default=None,
                            help="Feature transform in front of main network (in nnet format)")
    parser.add_option("--apply-log", action="store_true", dest="apply_log", default=False,
                            help="Transform MLP output to logscale (bool, default = false)")
    parser.add_option("--do-test", action="store_true", dest="do_test", default=False,
                            help="Do testing on the scp file (bool, default = false)")
    parser.add_option("--class-frame-counts", dest="class_frame_counts", default=None,
                            help="Vector with frame-counts of pdfs to compute log-priors.\
                            (priors are typically subtracted from \
                            log-posteriors or pre-softmax activations)")
    parser.add_option("--no-softmax", action="store_true", dest="no_softmax", default=False,
                            help="No softmax on MLP output (or remove it if found), \
                                    the pre-softmax activations will be used as \
                                    log-likelihoods, log-priors will be  \
                                    subtracted (bool, default = false)")
    parser.add_option("--prior-cutoff", dest="prior_cutoff", default=1e-10,
                            help="Classes with priors lower than cutoff \
                            will have 0 likelihood (float, default = 1e-10)")
    parser.add_option("--prior-scale", dest="prior_scale", default=1,
                            help="Scaling factor to be applied on \
                            pdf-log-priors (float, default = 1)")
    parser.add_option("--use-gpu", action="store_true", dest="use_gpu", default=False,
                            help="False,|True, Does nothing at the moment ")

    global learning_rate_val
    global delta 
    global delta_bar
    global E_valid

    print 'Learning rate', learning_rate_val

    ## TODO: Implement prior cutoff and class-frame-counts
    #if len(args) != 3:
    #    print usage
    #    exit(1)
    (options,args) = parser.parse_args(args=args)
    mlp_test = options.do_test 

    model = args[1]
    training_feat_file = args[2]
    dev_feat_file = args[3]
    training_alignment_dir = args[4]
    dev_alignment_dir = args[5]
    model_alignment = training_alignment_dir + '/final.mdl'
    adapted_model_path = args[6]
    adapted_model_name = args[7]

    if not os.path.exists(adapted_model_path):
        os.makedirs(adapted_model_path)

    number_of_classes = 0


# Read the model 
####################################################
## Create MLP class using the pre-initialized weights
####################################################
    model_txt = readKaldiDNN.convertModeltoText(model)
    model = readKaldiDNN.readMainModel(model_txt)
    W = []
    b = []
    activations = []

    print args
    print options

    for ele in model:
        # W.append(np.transpose(ele.W))
        W.append(np.array(ele.W))
        b.append(np.squeeze(ele.b))
        assert ele.activation in activation_dict, "Did not understand the activation type"
        activations.append(activation_dict[ele.activation])
    
    number_of_classes = len(b[-1])
    mlp = define_nn_preset(W, b, activations)
# if you do not want to load kaldi DBN, use this option where the wgts are randomly initialized
#    mlp = define_nn(440, 2048, 3, 1983)
####################################################

    train_interface_fid = h5py.File(training_feat_file)
    dev_interface_fid = h5py.File(dev_feat_file)

    all_training_keys = []
    all_dev_keys = []
    
    input_feat_count = 0
    valid_feat_count = 0

# Get alignment 
    train_feat_alignment = readKaldiData.readAlignments(model_alignment, training_alignment_dir)
    dev_feat_alignment = readKaldiData.readAlignments(model_alignment, dev_alignment_dir)

# Not all data has alignment information. Total data is obtained using the alignment size
    total_train_data = len(train_feat_alignment.align_dictionary)
    total_dev_data = len(dev_feat_alignment.align_dictionary)

    training_data_len = int(training_percentage * total_train_data)
    dev_data_len = int(dev_percentage * total_dev_data)

# Get all the keys in an array
    key_count = 0
    for key in train_interface_fid['feats']:
            key_count += 1
            if not key in train_feat_alignment.align_dictionary:
                print 'Alignments not found for:' + key 
                continue
            if key_count < training_data_len:
                all_training_keys.append(key)
                input_feat_count += train_interface_fid['feats'][key].shape[1]

    key_count = 0
    for key in dev_interface_fid['feats']:
            key_count += 1
            if not key in dev_feat_alignment.align_dictionary:
                print 'Alignments not found for:' + key 
                continue
            if key_count < dev_data_len:
                all_dev_keys.append(key)
                valid_feat_count += dev_interface_fid['feats'][key].shape[1]


    train_part = all_training_keys
    valid_part = all_dev_keys

    print("%d train examples" % len(train_part))
    print("%d valid examples" % len(valid_part))
    print("%d train features count " % int (input_feat_count))
    print("%d valid features count " % int (valid_feat_count))
    
                


# Mean shift and variance scaling    
    if not options.feature_transform is None:
        feat_trans_txt = readKaldiDNN.convertModeltoText(options.feature_transform)
        feat_trans = readKaldiDNN.readFeatTransform(feat_trans_txt)
        left_splicing = np.abs(feat_trans.splicing[0])
        right_splicing = np.abs(feat_trans.splicing[-1])
        shift = feat_trans.shift
        scale = feat_trans.scale
        shift = np.float32(np.array(shift))
        scale = np.float32(np.array(scale))
    else:
        sys.stderr.write("No transform file. Continue...")
        shift = np.zeros((W[0].shape[1],))
        scale = np.ones((W[0].shape[1],))
        shift = np.float32(np.array(shift))
        scale = np.float32(np.array(scale))

####################################################

    mlp_input  = T.matrix('mlp_input', dtype=theano.config.floatX)
    # mlp_output = T.matrix('mlp_output', dtype=theano.config.floatX)
    mlp_output = T.lvector('mlp_output')

    final_mlp_input = mlp_input
    # ipdb.set_trace()
    t_learning_rate_val = T.fscalar('lr')  

        
    grad_accums = []
    updt_accums = []
    for param in mlp.params:
        grad_accums.append(theano.shared(param.get_value()*0., broadcastable=param.broadcastable))
        updt_accums.append(theano.shared(param.get_value()*0., broadcastable=param.broadcastable))

    # MLP output        
    # cost = mlp.negative_log_likelihood(final_mlp_input)
    # supervised_cost = T.mean(mlp.XEnt(final_mlp_input, mlp_output)) + 1e-4 * mlp.sqr_L2_norm()
    supervised_cost = T.mean(mlp.XEnt(final_mlp_input, mlp_output)) 
    #updates = MLP.ADADELTA_updates(supervised_cost, mlp.params, grad_accums, updt_accums,rho=0.95)
    updates = MLP.NAG_updates(supervised_cost, mlp.params, grad_accums, t_learning_rate_val, momentum)
    train_model = theano.function(
            inputs=[final_mlp_input, mlp_output, t_learning_rate_val],
            outputs=supervised_cost,
            updates=updates,
            on_unused_input='warn')
    XEntError =  mlp.XEnt(final_mlp_input, mlp_output)
    XEntCost = theano.function(
            inputs=[final_mlp_input, mlp_output],
            outputs=XEntError,
            on_unused_input='warn')
    classification_accuracy =  mlp.classification_accuracy(final_mlp_input, mlp_output)
    classification_accuracy_fn = theano.function(
            inputs=[final_mlp_input, mlp_output],
            outputs=classification_accuracy,
            on_unused_input='warn')

    best_cost = np.Inf
    best_grad_accums = []

    def doValidation(DEV_MINI_EPOCH=DEV_MINI_EPOCH):
        print ('Doing validation')
        # Do validation
        kf = get_minibatches_idx(len(valid_part), DEV_MINI_EPOCH, shuffle=False)
        valid_cost = 0
        mask_cost = 0
        classification_accuracy_val = 0
        total_feats = 0
        for _, valid_index in kf:
            X = [(create_context(dev_interface_fid['feats'][valid_part[key_idx]][...])
                    + shift[...,None]) * scale[...,None]
                 for key_idx in valid_index]
            alignment = [ dev_feat_alignment.align_dictionary[valid_part[key_idx]] for key_idx in valid_index]
            X = np.float32(np.hstack(X))
            alignment = np.hstack(alignment)
            total_feats += X.shape[1]
            valid_cost += np.sum(XEntCost(X,np.int16(alignment)))
            classification_accuracy_val += np.sum(classification_accuracy_fn(X,np.int16(alignment.T)))
        
        valid_cost = valid_cost/total_feats
        classification_accuracy_val = np.float(classification_accuracy_val)/total_feats 
        classification_error = 1 - classification_accuracy_val
        return valid_cost, classification_error

    validation_counter = 0
    print 'Validation cost before training'
    XEntCost_val, classification_accuracy_error_val = doValidation()
    best_validation_cost_val = classification_accuracy_error_val

    print 'Validation cost:', XEntCost_val
    print 'Classification error', classification_accuracy_error_val

    E_valid[0] = XEntCost_val

    mini_batch_count = 0
    for iteration in range(MAX_EPOCH):
        start_str = time.strftime('%X %x %Z')
        print()
        print('-' * 50)
        print('Iteration', iteration)
        sys.stdout.flush()
        kf = get_minibatches_idx(len(train_part), MINI_EPOCH, shuffle=True)
        for _, train_index in kf:
            validation_counter += 1
            sys.stdout.write('.')
            print ("Batch number:", validation_counter)
            X = [(create_context(train_interface_fid['feats'][train_part[key_idx]][...] )
                    + shift[...,None]) * scale[...,None]
                   for key_idx in train_index]
            alignment = [ train_feat_alignment.align_dictionary[train_part[key_idx]] for key_idx in train_index]
            X = np.float32(np.transpose(np.hstack(X)))
            alignment = np.hstack(alignment)
            shuffled_index = np.arange(len(X), dtype="int32")
            np.random.shuffle(shuffled_index)

            X = X[shuffled_index]
            alignment = np.int16(alignment[shuffled_index])

            all_X_split = np.array_split(X,len(X)/mini_batch_size)
            all_align_split = np.array_split(alignment,len(alignment)/mini_batch_size)
            current_cost = 0

            print 'Learning rate', learning_rate_val
            ipdb.set_trace()

# Break the task into mini batches
            for X_mini,  align_mini in zip(all_X_split,  all_align_split):
                mini_batch_count += 1
                temp_cost = train_model(X_mini.T,np.transpose(align_mini), learning_rate_val)
                current_cost += temp_cost

            training_cost = current_cost/np.float(mini_batch_count)
            print "Training cost:", training_cost
            sys.stdout.flush()
            if not validation_counter % DO_VALIDATION_EVERY_N_MINI_EPOCH == 0:
                continue

            XEntCost_val, classif_cost_val = doValidation()

# Update the learning rate
            E_valid[1] = XEntCost_val
            delta[1] = (E_valid[1] - E_valid[0]) / E_valid[1]
            if delta[1]*delta_bar[0] < 0 and np.abs(delta_bar[0]) > theta_lr:
                learning_rate_val *= decrement_lr
            else:
                learning_rate_val *= increment_lr
            delta_bar[1] = alpha_lr*delta[1] + (1 - alpha_lr)*delta_bar[0]
            assert len(delta) == len(delta_bar) == len(E_valid) == 2
            delta[0] = delta[1]
            delta_bar[0] = delta_bar[1]
            E_valid[0] = E_valid[1]

            print 'XEnt cost:', XEntCost_val
            print 'Classification error', classif_cost_val

            if classif_cost_val < best_validation_cost_val:
                print("Saving Model...")
                mlp.save_model(adapted_model_path, adapted_model_name)
                best_validation_cost_val = classif_cost_val
                print '*'*50
            sys.stdout.flush()

if __name__=="__main__":
    main(sys.argv)
    
