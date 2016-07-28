import os
import sys
import numpy as np
import ipdb
import tempfile 

kaldiPath_new = "/talc/multispeech/calcul/users/ssivasankaran/softwares/kaldi"
kaldiPath_old = "/home/ssivasankaran/softwares/kaldi_nnet3_cpu"

class layer_def:

    def __init__(self):
        self.W = None
        self.b = None
        self.layerSize = None
        self.activation = None
        self.activation_size = None
    
    def setW(self,w):
        self.W = w

    def setB(self,b):
        self.b = b

    def setActivation(self,activation):
        self.activation = activation

    def setActivationSize(self,activation_size):
        self.activation_size = activation_size

    def setLayerSize(self,layer_size):
        self.layerSize = layer_size 

    def setParams(self, w, b,layer_size, activation_size, activation):
        self.b = np.asarray(b).reshape(len(b),1)
        self.activation = activation
        self.activation_size = activation_size
        self.W = w
        self.layerSize = layer_size

class feat_layer:
    def __init__(self):
        self.splicing = None
        self.shift = None
        self.scale = None
        self.dim = None

    def setParams(self, splicing, dim, shift, scale):
        self.splicing = splicing
        self.shift = shift
        self.scale = scale
        self.dim = dim


def convertModeltoText(modelPath, kaldi_ver=0):
    featFile = tempfile.mkstemp()
    if kaldi_ver == 1:
        kaldiPath =  kaldiPath_new
    else:
        kaldiPath =  kaldiPath_old
    os.system(kaldiPath+'/src/nnetbin/nnet-copy --binary=false ' + modelPath + ' ' + featFile[-1])
    return featFile[-1]


def readMainModel(modelFileName, kaldi_ver=0):    
    if not kaldi_ver == 0:
        return readMainModelKaldi2016(modelFileName)
    all_layers = []
    fid = open(modelFileName)
    junk = fid.readline() # Nnet
    line = fid.readline().strip() #  %<AffineTransform> XXX YYY
    while not line == '</Nnet>':
        size = [int(x) for x in line.strip().split()[1:]]
        line = fid.readline().strip() 
    
        layer = layer_def()
        w = []
        # print size
    
        for neuron in xrange(size[0]-1):
            line = fid.readline().strip()
            transform = [float(x) for x in line.split()]
            w.append(transform)
    
        line = fid.readline().strip()
        line = line.replace(']','')
        transform = [float(x) for x in line.split()]
        w.append(transform)
    
        line = fid.readline().strip()
        line = line.replace('[','')
        line = line.replace(']','')
        bias = [float(x) for x in line.split()]
        
        line = fid.readline().strip() 
        activation = line.split()[0].replace('<','').replace('>','')
        activation_size = [int(x) for x in line.strip().split()[1:]]
    
        layer.setParams(w,bias, size, activation_size, activation)
        line = fid.readline().strip() 
        all_layers.append(layer)

    # print len(all_layers)
    return all_layers

def readFeatTransform(fileName, kaldi_ver=0):
    if not kaldi_ver == 0:
        return readFeatTransformKaldi2016(fileName)
    fid = open(fileName)
    junk = fid.readline() # Nnet
    featLayer = feat_layer()

    line = fid.readline().strip() #  % splicing info
    dim = [int(x) for x in line.strip().split()[1:]]

    line = fid.readline().strip() #  % splicing info
    line = line.replace('[','').replace(']','')
    feat_splice_index = [int(x) for x in line.strip().split()]
    line = fid.readline().strip() #  % splicing info
    line = fid.readline().strip().split('[')[-1]
    line = line.replace(']','')
    shift = [float(x) for x in line.split()]

    line = fid.readline()
    line = fid.readline().strip().split('[')[-1]
    line = line.replace(']','')
    scale = [float(x) for x in line.split()]
    featLayer.setParams(feat_splice_index, dim, shift, scale)
    return featLayer

def readMainModelKaldi2016(modelFileName):    
    all_layers = []
    fid = open(modelFileName)
    junk = fid.readline() # Nnet
    line = fid.readline().strip() #  %<AffineTransform> XXX YYY
    while not line == '</Nnet>':
# Get the size
        size = [int(x) for x in line.strip().split()[1:]]
# <LearnRateCoeff>... 
        line = fid.readline() 
# [        
        line = fid.readline() 

        layer = layer_def()
        w = []

        for neuron in xrange(size[0]-1):
# Get the numbers
            line = fid.readline().strip()
            transform = [float(x) for x in line.split()]
            w.append(transform)
    
        line = fid.readline().strip()
        line = line.replace(']','')
        transform = [float(x) for x in line.split()]
        w.append(transform)
    
# Get the  bias
        line = fid.readline().strip()
        line = line.replace('[','')
        line = line.replace(']','')
        bias = [float(x) for x in line.split()]
        
# New kaldi has End component string. Remove that        
        line = fid.readline() 

# Get activation type
        line = fid.readline().strip() 
        activation = line.split()[0].replace('<','').replace('>','')
        activation_size = [int(x) for x in line.strip().split()[1:]]

# New kaldi has End component string after activation type. Remove that        
        line = fid.readline() 
    
        layer.setParams(w, bias, size, activation_size, activation)
        line = fid.readline().strip() 
        all_layers.append(layer)

    # print len(all_layers)
    return all_layers

def readFeatTransformKaldi2016(fileName):
    fid = open(fileName)
    junk = fid.readline() # Nnet
    featLayer = feat_layer()

    line = fid.readline().strip() #  % splicing info
    dim = [int(x) for x in line.strip().split()[1:]]

# Splicing index
    line = fid.readline().strip() 
    line = line.replace('[','').replace(']','')
    feat_splice_index = [int(x) for x in line.strip().split()]
# New kaldi has End component string. Remove that        
    line = fid.readline() 

# Get the add component
    line = fid.readline().strip() 
    line = fid.readline().strip().split('[')[-1]
    line = line.replace(']','')
    shift = [float(x) for x in line.split()]

# New kaldi has End component string. Remove that        
    line = fid.readline() 

# Get the scale component
    line = fid.readline()
    line = fid.readline().strip().split('[')[-1]
    line = line.replace(']','')
    scale = [float(x) for x in line.split()]
    featLayer.setParams(feat_splice_index, dim, shift, scale)
    return featLayer

# nnet_orig = "/talc_data3/parole/calcul/ssivasankaran/kaldi_exp/chime3_new/s5/exp/tri4a_mfcc_dnn_tr05_combined_ds_3_mc_real_fmllr/nnet/nnet_7.dbn_dnn_iter15_learnrate9.76565e-07_tr0.9404_cv1.8576"
# nnet_featTransform = "/talc_data3/parole/calcul/ssivasankaran/kaldi_exp/chime3_new/s5/exp/tri4a_mfcc_dnn_tr05_combined_ds_3_mc_real_fmllr/final.feature_transform"
# copy_nnet = kaldiPath + "/src/nnetbin/nnet-copy"
# nnet_txt_path =  convertModeltoText(nnet_orig)
# nnet_feat_trans_txt_path = convertModeltoText(nnet_featTransform)
# print "File in " + nnet_txt_path + " " + nnet_feat_trans_txt_path
# feat_tra = readFeatTransform(nnet_feat_trans_txt_path)
# dnn_model = readMainModel(nnet_txt_path)
