from optparse import OptionParser
import MLP
import theano
import readKaldiDNN 
import sys
import readKaldiData
import h5py
import numpy as np
import theano.tensor as T
import ipdb

#  nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=exp/reverb_test/puma_data/reverb/rt_elr/tri4a_dnn_tr_rt_0.0_0.45_elr_10.0_15.0/final.feature_transform --class-frame-counts=exp/reverb_test/puma_data/reverb/rt_elr/tri4a_dnn_tr_rt_0.0_0.45_elr_10.0_15.0/ali_train_pdf.counts --use-gpu=no exp/reverb_test/puma_data/reverb/rt_elr/tri4a_dnn_tr_rt_0.0_0.45_elr_10.0_15.0/final.nnet "ark,s,cs:copy-feats scp:data-fbank/reverb_test/puma_data/reverb/rt_elr/et_rt_0.0_0.45_elr_10.0_15.0/split4/1/feats.scp ark:- |" ark:-

RDMSEED = 111214

activation_dict = {}
activation_dict['Sigmoid'] = T.nnet.sigmoid
activation_dict['Relu'] = T.nnet.relu
activation_dict['Softmax'] = T.nnet.softmax

def load_kaldi_priors(path, prior_cutoff, uniform_smoothing_scaler=0.05):
    assert 0 <= uniform_smoothing_scaler <= 1.0, (
        "Expected 0 <= uniform_smoothing_scaler <=1, got %f" % uniform_smoothing_scaler
    )
    numbers = np.fromregex(path, r"([\d\.e+]+)", dtype=[('num', np.float32)])
    class_counts = np.asarray(numbers['num'], dtype=theano.config.floatX)
    # compute the uniform smoothing count
    uniform_priors = np.ceil(class_counts.mean() * uniform_smoothing_scaler)
    priors = (class_counts + uniform_priors) / class_counts.sum()
    #floor zeroes to something small so log() on that will be different from -inf or better skip these in contribution at all i.e. set to -log(0)?
    priors[priors < prior_cutoff] = prior_cutoff
    assert np.all(priors > 0) and np.all(priors <= 1.0), (
        "Prior probabilities outside [0,1] range."
    )
    log_priors = np.log(priors)
    assert not np.any(np.isinf(log_priors)), (
        "Log-priors contain -inf elements."
    )
    return log_priors

def define_nn_preset(W_init, b_init, activations):
    return MLP.MLP(W_init, b_init, activations)

def define_nn(n_input_dims, n_hidden_dims, n_hidden_layer, n_output_dims, W_init=None, b_init=None):
    layer_sizes = [ n_input_dims ]
    activations = []
    for i in xrange(0, n_hidden_layer):
        layer_sizes.append(n_hidden_dims)
        activations.append(T.nnet.sigmoid)

    layer_sizes.append(n_output_dims)
    # activations.append(T.nnet.relu)
    activations.append(None)

    # Set initial parameter values
    act_count = 0
    if W_init == None:
        W_init = []
        b_init = []
        for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
            if activations[act_count] == T.tanh:
                W_init.append(np.random.RandomState(RDMSEED).uniform(
                    low=-np.sqrt(6. / (n_output + n_input)), 
                    high=np.sqrt(6. / (n_output + n_input)),
                    size=(n_output, n_input) ) )
            elif activations[act_count] == T.nnet.sigmoid:
                W_init.append(np.random.RandomState(RDMSEED).uniform(
                    low=-4*np.sqrt(6. / (n_output + n_input)), 
                    high=4*np.sqrt(6. / (n_output + n_input)),
                    size=(n_output, n_input) ) )
            elif activations[act_count] == T.nnet.relu:
                W_init.append(np.random.RandomState(RDMSEED).normal(0., np.sqrt(2./n_input), (n_output, n_input)))
            else:
                W_init.append(np.random.RandomState(RDMSEED).normal(0., 0.01, (n_output, n_input)))
            act_count += 1
            b_init.append(np.zeros(n_output))
    
        #mlp = MLP.MLP_wDO(W_init, b_init, activations)
        mlp = MLP.MLP(W_init, b_init, activations)
    else:
        mlp = MLP.MLP(W_init, b_init, activations)

    return mlp


def main(args=None):
    usage = args[0] + " [options] <model-dir> <feat.scp>"
    parser = OptionParser(usage)
    parser.add_option("--apply-log", action="store_true", dest="apply_log", default=False,
                            help="Transform MLP output to logscale (bool, default = false)")
    parser.add_option("--class-frame-counts", dest="class_frame_counts", default=None,
                            help="Vector with frame-counts of pdfs to compute log-priors. (priors are typically subtracted from log-posteriors or pre-softmax activations)")
    parser.add_option("--feature-transform", dest="feature_transform", default=None,
                            help="Feature transform in front of main network (in nnet format)")
    parser.add_option("--no-softmax", action="store_true", dest="no_softmax", default=False,
                            help="No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted (bool, default = false)")
    parser.add_option("--prior-cutoff", dest="prior_cutoff", default=1e-10,
                            help="Classes with priors lower than cutoff will have 0 likelihood (float, default = 1e-10)")
    parser.add_option("--prior-scale", dest="prior_scale", default=1,
                            help="Scaling factor to be applied on pdf-log-priors (float, default = 1)")
    parser.add_option("--use-gpu", action="store_true",dest="use_gpu", default=False,
                            help="False,|True, only has effect if compiled with CUDA")


    ## TODO: Implement prior cutoff and class-frame-counts
    #if len(args) != 3:
    #    print usage
    #    exit(1)
    (options,args) = parser.parse_args(args=args)
        

    feature_file = args[1]
    model = args[2]


    if options.apply_log and options.no_softmax:
        sys.stderr.write("Apply log and no softmax cannot be true at the same time")
        exit(1)

    # Load the priors
    if not options.class_frame_counts == None:
        class_prior = load_kaldi_priors(options.class_frame_counts, options.prior_cutoff)

    left_splicing = 0
    right_splicing = 0
    mlp_final_output = ""
#    model_txt = readKaldiDNN.convertModeltoText(model)
#    model = readKaldiDNN.readMainModel(model_txt)

    fid = h5py.File(model)
    W = []
    b = []
    activations = []
    layer_size = len(fid['params'].keys())
    for count in range(layer_size):
            W.append(fid['params/' + str(count) + '/W'][...])
            b.append(np.squeeze(fid['params/' + str(count) + '/b'][...]))
            activations.append(T.nnet.sigmoid)

#    for ele in model:
#        # W.append(np.transpose(ele.W))
#        W.append(np.array(ele.W))
#        b.append(np.squeeze(ele.b))
#        assert ele.activation in activation_dict, "Did not understand the activation type"
#        activations.append(activation_dict[ele.activation])
    
    # Remove the last Softmax layer if needed
    if options.no_softmax :
        activations[-1] = None

    mlp = define_nn_preset(W,b,activations)
    # mlp = define_nn(440, 2048, 7, 1983)
                
    if not options.feature_transform == None:
        feat_trans_txt = readKaldiDNN.convertModeltoText(options.feature_transform)
        feat_trans = readKaldiDNN.readFeatTransform(feat_trans_txt)
        left_splicing = np.abs(feat_trans.splicing[0])
        right_splicing = np.abs(feat_trans.splicing[-1])
        shift = feat_trans.shift
        scale = feat_trans.scale
    else:
        sys.stderr.write("No transform file. Continue...")

    features = readKaldiData.readFeatures(feature_file, splicing_right=right_splicing, splicing_left=left_splicing)

    mlp_input  = T.matrix('mlp_input', dtype=theano.config.floatX)
    mlp_output = T.matrix('mlp_output', dtype=theano.config.floatX)

    final_mlp_input = mlp_input
    # ipdb.set_trace()

    # Do mean and variance normalization if required
    if not options.feature_transform == None:
        final_mlp_input = ( mlp_input + np.array(shift)[...,None] ) * np.array(scale)[...,None]
        
    # MLP output        
    tr_dnn_output = mlp.output(final_mlp_input)
    test_model = theano.function(inputs=[mlp_input], outputs=tr_dnn_output)

    while True:
        feats = np.float32(features.getNextWavData())
        if features.done and len(feats) == 0:
            break
        wavID = features.wavID
        prob_values = np.transpose(test_model(np.transpose(feats)))
        if options.apply_log:
            prob_values = np.log(prob_values + np.finfo(float).eps)
        # Include the priors            
        if not options.class_frame_counts == None:
            prob_values = prob_values - class_prior
        prob_values = np.around(prob_values, decimals=5)
        mlp_final_output = wavID + " [\n"
        mlp_final_output += "".join(map(str,prob_values.tolist())).replace("]["," \n ").replace("[","").replace("]","\n]").replace(","," ")

        # Put it out
        sys.stdout.write(mlp_final_output)
        sys.stdout.flush()

if __name__=="__main__":
    main(sys.argv)
    
