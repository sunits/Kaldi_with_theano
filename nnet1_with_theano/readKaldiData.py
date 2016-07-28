import tempfile
import os
import sys
import re
import numpy as np


kaldiPath = "/talc/multispeech/calcul/users/ssivasankaran/softwares/kaldi"

class readAlignments:

    def __init__(self, model, align_path):
        self.model = model
        self.align_path = align_path
        self.align_dictionary = {}
        tempFile = tempfile.mkstemp()
            
        os.system('ali-to-pdf ' + model +' \"ark:gunzip -c ' + align_path + '/ali.*.gz |\" ark,t:' + tempFile[-1])
        with open(tempFile[-1]) as fid:
            for line in fid:
                line_split = line.strip().split()
                self.align_dictionary[line_split[0]] = [int(x) for x in line_split[1:]]

class readFeatures:

    def __init__(self, scp_file, splicing_right=0, splicing_left=0):
        self.featFile = tempfile.mkstemp()
        os.system('copy-feats --binary=false scp:' + scp_file + \
                 ' ark:- | splice-feats --left-context=' + str(splicing_left) + \
                 ' --right-context=' + str(splicing_right) + ' ark:- ark,t:' + self.featFile[-1] )
        self.fid = open(self.featFile[-1])
        self.done = False

    def lineToVector(self, featEntry):
           return [float(feat_value) for feat_value in featEntry.strip().split() 
                if  re.search('[0-9]+\.*[0-9]*', feat_value)]

    def getNextWavData(self):
        if self.done:
            return False
        index = 0
        featureHolder = []
        for featEntry in self.fid:
            if ']' in featEntry:
                featureHolder.append(self.lineToVector(featEntry))
                # If the line ends with "...]", then the last line will not contain EOF
                break
            elif not '[' in featEntry:
                featureHolder.append(self.lineToVector(featEntry))
            elif '[' in featEntry:
                self.wavID = featEntry.strip().split('[')[0].strip()
        else:
            self.done = True
        return featureHolder

    def done(self):
        self.fid.close()
    
    def restart(self):
        self.fid.seek(0)
        self.done = False

    def __del__(self):
        self.fid.close()

class readPDFCount:

    def __init__(self,file_path):
        self.file_path = file_path
        with open(file_path) as fid:
            for ele in fid:
                ele = ele.strip().replace('[','').replace(']','')
                self.count = [ int(x) for x in ele.strip().split()]
        self.prior = np.divide(self.count, float(np.sum(self.count)))

class otherIO:

    def __init__(self):
        pass

    def writeKaldiMatrixFile(self,file_path, wav_id, data):
# data should be in the format [feat x likelihood]
        with open(file_path, 'w') as wid:
            wid.writelines(wav_id + '  [\n  ')
            for row in data:
                for ele in row:
                    wid.writelines(str(ele)+' ')
                wid.writelines('\n')
            else:
                wid.writelines(']')

class kaldiFunctionlity:

    def __init__(self):
        self.latgenFasterMappedOptions = {}
        self.latgenFasterMappedOptions['--min-active'] = 200 
        self.latgenFasterMappedOptions['--max-active'] = 7000 
        self.latgenFasterMappedOptions['--max-mem'] = 50000000
        self.latgenFasterMappedOptions['--beam'] = 18.0
        self.latgenFasterMappedOptions['--lattice-beam'] = 10.0 
        self.latgenFasterMappedOptions['--acoustic-scale'] = 0.10 
        self.latgenFasterMappedOptions['--allow-partial'] = "true"

        self.latticeScaleOptions = {}
        self.latticeScaleOptions['--inv-acoustic-scale'] = 4

    def latgenFasterMapped(self,options, mdl_file, hclg, rSpecifier, wSpecifier):
        for ele in options:
            self.latgenFasterMappedOptions[ele] = options[ele]
        self.cmd = "latgen-faster-mapped "             

        for ele in self.latgenFasterMappedOptions:
            self.cmd += ele + '=' + self.latgenFasterMappedOptions[ele] + ' '

        self.cmd += mdl_file + ' ' + hclg + ' ark,t:' + rSpecifier 
        self.cmd += ' \"ark:|gzip -c >' + wSpecifier + '\"'
        print self.cmd
        os.system(self.cmd)

    def LatGz2Trans(self, options, lat_file, WList, outFile):
        for ele in options:
            self.latticeScaleOptions[ele] = options[ele]
        self.cmd = "lattice-scale "
        
        for ele in self.latticeScaleOptions:
            self.cmd += str(ele) + '=' + str(self.latticeScaleOptions[ele]) + ' '
       
        self.cmd += '\"ark:gunzip -c ' + lat_file + '|\" ark:- | lattice-add-penalty '
        self.cmd += '--word-ins-penalty=0.0 ark:- ark:- | lattice-best-path --word-symbol-table=' + WList 
        self.cmd += ' ark:- ark,t:' + outFile
        print self.cmd 
        os.system(self.cmd)

class readTree:
    def __init__(self,path):
        self.path = path
        self.treeFile = tempfile.mkstemp()
        os.system('tree-info ' + self.path + ' > '  + self.treeFile[-1])
        self.info = {}
        with open(self.treeFile[-1]) as fid:
            for ele in fid:
                text, information = ele.strip().split()
                self.info[text] = int(information)
        os.system('rm '+self.treeFile[-1])                
