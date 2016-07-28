echo `hostname`
GPU='gpu0'

# Trained RBM
DNN_BASEPATH='/talc/multispeech/calcul/users/ssivasankaran/kaldi_exp/chime3_new/s5/exp/reverb_test/puma_data/reverb/clean_align/layers/3/tri4a_dnn_pretrain_tr05_multi_reverbed/'

# Features in hdf5 file
train_feats='/talc/multispeech/calcul/users/ssivasankaran/kaldi_exp/chime3_new/s5/data-fbank/reverb_test/puma_data/reverb/tr05_multi_reverbed/feats_no_splice.hdf5'
valid_feats='/talc/multispeech/calcul/users/ssivasankaran/kaldi_exp/chime3_new/s5/data-fbank/reverb_test/puma_data/reverb/dt05_multi_reverbed/feats_no_splice.hdf5'
valid_align='/talc/multispeech/calcul/users/ssivasankaran/kaldi_exp/chime3_new/s5/exp/tri3b_tr05_orig_clean_ali_dt05'
train_align='/talc/multispeech/calcul/users/ssivasankaran/kaldi_exp/chime3_new/s5/exp/reverb_test/tri3b_tr05_orig_clean_ali'


model="$DNN_BASEPATH/3.dbn"
feat_transform="$DNN_BASEPATH/final.feature_transform"
class_priors="$DNN_BASEPATH/ali_train_pdf.counts"

# Path to save the models
model_path='models/dummy'
# Name of the models
model_name='dummy.hdf5'
mkdir -p $model_path


# Keep a local copy and avoid reading from NFS
# If you are not using NFS you can disable it, Make sure the temp paths are set to the original paths
temp_train_feats='/tmp/tr05_multi_reverbed_feats.hdf5'
temp_valid_feats='/tmp/dt05_multi_reverbed_feats.hdf5'

if [[ ! -e $temp_train_feats ]]; then
    cp $train_feats $temp_train_feats
fi

if [[ ! -e $temp_valid_feats ]]; then
    cp $valid_feats $temp_valid_feats
fi

THEANO_FLAGS="floatX=float32,device=$GPU,lib.cnmem=0.95" python nnet_train.py --feature-transform $feat_transform \
                                                                                $model \
                                                                                $temp_train_feats \
                                                                                $temp_valid_feats  \
                                                                                $train_align \
                                                                                $valid_align \
                                                                                $model_path \
                                                                                $model_name
