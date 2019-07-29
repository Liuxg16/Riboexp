import sys, os
from os.path import join,dirname
import re, fileinput, math
import numpy as np
import random, copy
import cPickle as pickle
import string
#import h5py
#import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
#import pandas as pd

class Dataset_yeast_structure(object):

    def __init__(self, rnd_seed=1234,testrate = 0.3,vocab=None, relative_offset = 0,\
            path = None):
        print('building the dataset')
        self.path = path
        self.rnd_seed = rnd_seed
        self.genVocab()
        traindata, trainnt, trainfold, trainlabels,\
            validdata, validnt, validfold, validlabels,\
            testdata, testnt, testfold, testlabels  = self.readfile()
        self.window_condon = relative_offset

        self.testnt, self.testdata, self.testfold, self.testlabels = testnt, testdata, testfold, testlabels
        
        res = [self.word2index(x, self.codon_vocab) for x in [traindata, validdata,testdata]]
        traindata, validdata,testdata = res[0], res[1], res[2]

        res = [self.word2index(x, self.char_vocab) for x in [trainnt, validnt,testnt]]
        trainnt, validnt,testnt = res[0], res[1], res[2]

        # res = [self.word2index(x, self.structure_vocab) for x in [trainfold, validfold,testfold]]
        # trainfold, validfold,testfold = res[0], res[1], res[2]

        self.train_data, self.train_nt, self.train_fold,self.train_labels =\
            process_structure(traindata, trainnt, trainfold,trainlabels,self.window_condon)

        self.valid_data, self.valid_nt, self.valid_fold,self.valid_labels =\
            process_structure(validdata, validnt, validfold,validlabels,self.window_condon)

        self.test_data, self.test_nt, self.test_fold,self.test_labels =\
            process_structure(testdata, testnt, testfold,testlabels,self.window_condon)


        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_data)
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_nt)
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_fold)
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_labels)

    def readfile(self):
        energypath = join('/'.join(self.path.split('/')[:-1]),'energy/energy.fold')
        docs_= open(energypath,"rb").read()
        docs = docs_.strip().split('\n')
        n_sample = int(len(docs)/3)
        name2folddict = {}
        name2seqdict = {}
        lastname = 0
        for i in range(n_sample):
            name = ''.join(docs[3*i].split()[0])
            text = docs[3*i+1].strip().replace('U','T')
            fold = float(docs[3*i+2].strip().split()[-1].strip(')').strip('('))
            if name2seqdict.has_key(name):
                assert lastname == name
                name2seqdict[name] += text[-1]
                name2folddict[name].append(fold)
            else:
                lastname = name
                name2seqdict[name] = text
                name2folddict[name] = [-1]*14+[fold]

        traindocs_= open(join(self.path,'traindata.txt'),"rb").read()
        docs = traindocs_.strip().split('\n')
        traindata = []
        trainlabels = []
        trainnt = []
        trainfold = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            traindata.append(d)
            trainlabels.append(label)
            trainnt.append(nt)

            foldtext = name2seqdict[name]
            foldsource = name2folddict[name]+[-1]*14
            index = string.index(foldtext, text)
            assert index == 13
            fold = foldsource[index:index+len(nt)]
            assert len(nt) ==len(fold)
            trainfold.append(fold)

        validdocs_= open(join(self.path,'validdata.txt'),"rb").read()
        docs = validdocs_.strip().split('\n')
        validdata = []
        validlabels = []
        validnt = []
        validfold = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            validdata.append(d)
            validlabels.append(label)
            validnt.append(nt)

            foldtext = name2seqdict[name]
            foldsource = name2folddict[name]+[-1]*14
            index = string.index(foldtext, text)
            assert index == 13
            fold = foldsource[index:index+len(nt)]
            assert len(nt) ==len(fold)
            validfold.append(fold)


        testdocs_= open(join(self.path,'testdata.txt'),"rb").read()
        docs = testdocs_.strip().split('\n')
        testdata = []
        testlabels = []
        testnt = []
        testfold = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            testdata.append(d)
            testlabels.append(label)
            testnt.append(nt)

            foldtext = name2seqdict[name]
            foldsource = name2folddict[name]+[-1]*14
            index = string.index(foldtext, text)
            assert index == 13
            fold = foldsource[index:index+len(nt)]
            assert len(nt) ==len(fold)
            testfold.append(fold)


        # print('The number of training genes is {}'.format(len(trainnt)))
        # print('The number of validation genes is {}'.format(len(validnt)))
        # print('The number of testing genes is {}'.format(len(testnt)))

        assert(len(trainlabels)==len(trainnt))
        assert(len(validlabels)==len(validnt))
        assert(len(testlabels)==len(testnt))

        return traindata, trainnt, trainfold, trainlabels,\
            validdata, validnt, validfold, validlabels,\
            testdata, testnt, testfold, testlabels

    def genVocab(self):
        self.char_vocab = {'A':1,'C':2,'G':3, 'T':4}
        self.codon_vocab = {}
        num = 0
        for c in self.char_vocab:
            for c1 in self.char_vocab:
                for c2 in self.char_vocab:
                    num +=1
                    self.codon_vocab[c+c1+c2] = num
        
        temp = '(){}.|,'
        self.structure_vocab = {word:id+1 for id, word in enumerate(temp)}

    def getIndex(self,word,vocab):
        if vocab.has_key(word):
            return vocab[word]
        else:
            return vocab['unk']

    def word2index(self, docs, vocab):
        #docs = [' '.join(line) for line in docs]
        index_docs = [[self.getIndex(char,vocab) for char in doc] for doc in docs]
        # max_len = max([len(doc) for doc in index_docs])
        # index_docs = [doc+[vocab['mask']]*(max_len - len(doc)) for doc in index_docs]
        # index_docs = np.array(index_docs)
        return index_docs

    def getData(self):
        return self.train_data, self.train_nt, self.train_fold, self.train_labels,\
            self.valid_data, self.valid_nt, self.valid_fold, self.valid_labels,\
            self.test_data, self.test_nt, self.test_fold, self.test_labels

    def getVocab(self):
        return self.vocab

class Dataset_yeast(object):

    def __init__(self, rnd_seed=1234,testrate = 0.3,vocab=None, relative_offset = 0,\
            path = None):
        print('building the dataset')
        self.path = path
        self.rnd_seed = rnd_seed
        self.genVocab()
        traindata, trainnt, trainlabels,\
            validdata, validnt,  validlabels,\
            testdata, testnt, testlabels  = self.readfile()
        self.window_condon = relative_offset

        self.testnt, self.testdata,  self.testlabels = testnt, testdata, testlabels
        
        res = [self.word2index(x, self.codon_vocab) for x in [traindata, validdata,testdata]]
        traindata, validdata,testdata = res[0], res[1], res[2]

        res = [self.word2index(x, self.char_vocab) for x in [trainnt, validnt,testnt]]
        trainnt, validnt,testnt = res[0], res[1], res[2]


        self.train_data, self.train_nt,self.train_labels =\
            process_fragment(traindata, trainnt, trainlabels,self.window_condon)

        self.valid_data, self.valid_nt,self.valid_labels =\
            process_fragment(validdata, validnt, validlabels,self.window_condon)

        self.test_data, self.test_nt, self.test_labels =\
            process_fragment(testdata, testnt, testlabels,self.window_condon)


        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_data)
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_nt)
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.train_labels)

    def readfile(self):
        traindocs_= open(join(self.path,'traindata.txt'),"rb").read()
        docs = traindocs_.strip().split('\n')
        traindata = []
        trainlabels = []
        trainnt = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            traindata.append(d)
            trainlabels.append(label)
            trainnt.append(nt)


        validdocs_= open(join(self.path,'validdata.txt'),"rb").read()
        docs = validdocs_.strip().split('\n')
        validdata = []
        validlabels = []
        validnt = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            validdata.append(d)
            validlabels.append(label)
            validnt.append(nt)


        testdocs_= open(join(self.path,'testdata.txt'),"rb").read()
        docs = testdocs_.strip().split('\n')
        testdata = []
        testlabels = []
        testnt = []
        n_sample = int(len(docs)/3)
        for i in range(n_sample):
            name = docs[3*i].split()[0]
            label = [float(x) for x in docs[3*i+2].strip().split()]
            text = docs[3*i+1].strip()
            d = [text[3*i:3*i+3] for i in range(len(label))]
            nt = [x for x in text]
            assert len(d) == len(label) 
            testdata.append(d)
            testlabels.append(label)
            testnt.append(nt)


        assert(len(trainlabels)==len(trainnt))
        assert(len(validlabels)==len(validnt))
        assert(len(testlabels)==len(testnt))

        return traindata, trainnt, trainlabels,\
            validdata, validnt, validlabels,\
            testdata, testnt, testlabels

    def genVocab(self):
        self.char_vocab = {'A':1,'C':2,'G':3, 'T':4}
        self.codon_vocab = {}
        num = 0
        for c in self.char_vocab:
            for c1 in self.char_vocab:
                for c2 in self.char_vocab:
                    num +=1
                    self.codon_vocab[c+c1+c2] = num
        
        temp = '(){}.|,'
        self.structure_vocab = {word:id+1 for id, word in enumerate(temp)}

    def getIndex(self,word,vocab):
        if vocab.has_key(word):
            return vocab[word]
        else:
            return vocab['unk']

    def word2index(self, docs, vocab):
        #docs = [' '.join(line) for line in docs]
        index_docs = [[self.getIndex(char,vocab) for char in doc] for doc in docs]
        # max_len = max([len(doc) for doc in index_docs])
        # index_docs = [doc+[vocab['mask']]*(max_len - len(doc)) for doc in index_docs]
        # index_docs = np.array(index_docs)
        return index_docs

    def getData(self):
        return self.train_data, self.train_nt, self.train_labels,\
            self.valid_data, self.valid_nt, self.valid_labels,\
            self.test_data, self.test_nt, self.test_labels

    def getVocab(self):
        return self.vocab

def process_structure(docs_condons,docs_nt, docs_fold, docs_counts,  window_condon):

    offset_relative = int((window_condon-1)/2)
    offset_l =  offset_relative+int(window_condon-1)%2
    offset_r = offset_relative

    data = []
    nts = []
    folds = []
    labels = []

    genenum = 0
    for condon_seq,nt_seq,fold_seq, asite_seq1 in zip(docs_condons,docs_nt,docs_fold, docs_counts):
        assert len(condon_seq)==len(asite_seq1)
        assert 3*len(condon_seq)==len(nt_seq)
        n_seq = len(condon_seq)
        value_t = np.array(asite_seq1)
        # minv = np.log(np.min(value_t[value_t!=0]))
        # fold_seq = list(np.array(fold_seq)/np.sum(np.array(fold_seq)))
        # asite_seq = np.log(1+value_t.astype(float))
        asite_seq = value_t.astype(float)
        asite_sum = np.sum(asite_seq)
        if asite_sum<200:
            continue
        genenum += 1
        n_valid_asite = np.sum(asite_seq>0.5)
        asite_seq = asite_seq/(asite_sum/n_valid_asite)  # sum
        # asite_seq = (asite_seq-minv)/(np.max(asite_seq)-minv)  # min-max
        for i in range(n_seq):
            footprints = asite_seq[i]
            if footprints>1e-6:
                start = (i-offset_l)
                end = i+offset_r+1
                if start <0:
                    continue
                    #start = 0
                if end > n_seq:
                    continue
                    # end = n_seq
                datum = [x for x in condon_seq[start:end]]
                data.append(datum)
                nts.append(nt_seq[3*start:3*end])
                middel = int((3*(start+end)+1)/2)
                # folds.append(fold_seq[middel:3+middel])
                folds.append(fold_seq[3*start:3*end])
                labels.append(footprints)

    assert len(data)==len(labels)

    data = np.array(data).astype(int)
    ntdata = np.array(nts).astype(int)
    folddata = np.array(folds).astype(float)
    labels = np.array(labels)
   
    return data,ntdata, folddata, labels

def process_fragment(docs_condons,docs_nt, docs_counts,  window_condon):

    offset_relative = int((window_condon-1)/2)
    offset_l =  offset_relative+int(window_condon-1)%2
    offset_r = offset_relative

    data = []
    nts = []
    labels = []
    for condon_seq,nt_seq, asite_seq1 in zip(docs_condons,docs_nt, docs_counts):
        assert len(condon_seq)==len(asite_seq1)
        assert 3*len(condon_seq)==len(nt_seq)
        n_seq = len(condon_seq)
        value_t = np.array(asite_seq1)
        # minv = np.log(np.min(value_t[value_t!=0]))
        # asite_seq = np.log(1+value_t.astype(float))
        asite_seq = value_t.astype(float)
        asite_sum = np.sum(asite_seq)
        n_valid_asite = np.sum(asite_seq>0.5)
        asite_seq = asite_seq/(asite_sum/n_valid_asite)  # sum
        # asite_seq = (asite_seq-minv)/(np.max(asite_seq)-minv)  # min-max
        for i in range(n_seq):
            footprints = asite_seq[i]
            if footprints>1e-6:
                start = (i-offset_l)
                end = i+offset_r+1
                if start <0:
                    continue
                    #start = 0
                if end > n_seq:
                    continue
                    # end = n_seq
                datum = [x for x in condon_seq[start:end]]
                data.append(datum)
                nts.append(nt_seq[3*start:3*end])
                middel = int((3*(start+end)+1)/2)
                labels.append(footprints)

    assert len(data)==len(labels)

    data = np.array(data).astype(int)
    ntdata = np.array(nts).astype(int)
    labels = np.array(labels)
   
    return data,ntdata,  labels


if __name__ == "__main__":

    pass
