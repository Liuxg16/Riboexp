# coding=utf-8
# python2
import  sys, time, os , random, math
import cPickle as pickle, json
import numpy as np
import scipy.stats
from collections import Counter
from os.path import dirname, join
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
sys.path.append("..")
from models import *
from nntools.utils import say, mylog
from dataio import  Dataset_yeast_structure, Dataset_yeast
import  options

def main(args,kwargs):

    kwargs['data_path'] = args.data_path
    kwargs['drate'] = args.drate
    kwargs['n_hids'] = args.n_hids
    kwargs['mark'] = args.mark
    kwargs['optim'] = args.optim
    kwargs['window'] = args.window
    kwargs['lambda1'] = args.lambda1
    kwargs['parallel'] = args.parallel
    kwargs['seed'] = args.seed
    kwargs['L'] = args.L
    kwargs['weight_decay'] = args.weight_decay
    kwargs['clip_grad'] = args.clip_grad
    kwargs['max_norm'] = args.max_norm
    kwargs['fold'] = args.fold
    kwargs['no_structure'] = args.no_structure

    rnd_seed = args.seed
    random.seed(rnd_seed)
    torch.manual_seed(1111)

    assert args.data_path is not None
    if kwargs['fold']==0:
        dirpath =  join(args.data_path,'fold0')
    elif kwargs['fold']==1:
        dirpath =  join(args.data_path,'fold1')
    elif kwargs['fold']==2:
        dirpath =  join(args.data_path,'fold2')
    if args.no_structure:
        dataset = Dataset_yeast(1234, 1.0/3, relative_offset=args.window, path=dirpath)
        train_data,train_nt, train_labels, valid_data,valid_nt,\
            valid_labels, test_data, test_nt,test_labels = dataset.getData()
    else:
        dataset = Dataset_yeast_structure(1234, 1.0/3, relative_offset=args.window, path=dirpath)
        train_data,train_nt, train_fold, train_labels, valid_data,valid_nt, valid_fold,\
            valid_labels, test_data, test_nt, test_fold,test_labels = dataset.getData()
    vocab = dataset.codon_vocab
    print('building the model...')
    tmodel = TrainingModel(model_name= RiboexpModel, vocab=vocab, dataset = dataset,  **kwargs)

    if args.load is  not None: 
        print('loading the model...')
        with open(args.load, 'rb') as f:
            tmodel.model.load_state_dict(torch.load(f))
    if args.load0 is  not None: 
        with open(args.load0, 'rb') as f:
            tmodel.model1.load_state_dict(torch.load(f))
    if args.load1 is  not None: 
        with open(args.load1, 'rb') as f:
            tmodel.model2.load_state_dict(torch.load(f))
    if args.load2 is  not None: 
        with open(args.load2, 'rb') as f:
            tmodel.model3.load_state_dict(torch.load(f))

    if args.mode==0:
        if args.no_structure:
            mse, corr = tmodel.loop_train([train_data,train_nt,train_labels],\
                [valid_data,valid_nt, valid_labels],\
                [test_data, test_nt, test_labels])
        else:
            mse, corr = tmodel.loop_train([train_data,train_nt,train_fold,train_labels],\
                [valid_data,valid_nt, valid_fold,valid_labels],\
                [test_data, test_nt, test_fold,test_labels])
        return mse, corr

    elif args.mode == 1:
        assert args.load is not None, 'Please load a trained model!'
        print('=' * 89)
        print('-'*30+'Testing Performance'+'-'*30)
        print('=' * 89)
        if args.no_structure:
            test_data = tmodel.gen_batches([test_data,test_nt,test_labels])
        else:
            test_data = tmodel.gen_batches([test_data,test_nt, test_fold,test_labels])
        test_loss, test_mse, test_reward, corr = tmodel.validation(test_data, tmodel.model)
        print(' loss {:5.3f} | reward {:5.3} | MSE {:5.5}| Pearson\'s r {:3.3f}'.\
                format(test_loss, test_reward,test_mse,corr))
        return test_mse, corr

    elif args.mode == 2:

        assert args.load0 is not None, 'Please load THREE trained model!'
        assert args.load1 is not None, 'Please load THREE trained model!'
        assert args.load2 is not None, 'Please load THREE trained model!'


        print('=' * 89)
        print('-'*30+'Testing Performance'+'-'*30)
        print('=' * 89)
        if args.no_structure:
            test_data = tmodel.gen_batches([test_data,test_nt,test_labels])
        else:
            test_data = tmodel.gen_batches([test_data,test_nt, test_fold,test_labels])

        if args.fold==0:
            test_loss, test_mse, test_reward, corr = tmodel.validation(test_data, tmodel.model1)
        elif args.fold==1:
            test_loss, test_mse, test_reward, corr = tmodel.validation(test_data, tmodel.model2)
        elif args.fold==2:
            test_loss, test_mse, test_reward, corr = tmodel.validation(test_data, tmodel.model3)
        print(' fold {} | loss {:5.3f} | reward {:5.3} | MSE {:5.5}| Pearson\'s r {:3.3f}'.\
                format(args.fold, test_loss, test_reward,test_mse,corr))
        return test_mse, corr

    elif args.mode == 3:
        assert args.no_structure 
        tmodel.model_pre.n_repeat = 1
        print('=' * 89)
        print('-'*10+'Rescaled Ribosome Density Predictions for a Particular Genome'+'-'*10)
        print('=' * 89)

        data = open(args.gene_path,"r").read()
        seq = data.strip().split('\n')[0]
         
        preds = tmodel.testSeq(seq)
        print('| Position | Codon | Density |')
        for i,x in enumerate(preds):
            if preds[i]==0:
                ans = None
            else:
                ans = preds[i]
            print('| {:^8} | '.format(i)+' '+seq[3*i:3*i+3]+ ' '+' | {:^7}'.format(ans))

class TrainingModel(object):

    def __init__(self, model_name, vocab, dataset, **kwargs):
        self.args = args
        self.lr = args.lr
        self.eval = False
        self.kwargs = kwargs
        vocab = dataset.codon_vocab
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.best_val_value = None
        self.device_ids = kwargs['device_ids']
        ###############################################################################
        # Build the model
        ###############################################################################
        n_labels = 2

        self.model =  model_name(n_labels,  **kwargs)
        self.model1 = model_name(n_labels,  **kwargs)
        self.model2 = model_name(n_labels,  **kwargs)
        self.model3 = model_name(n_labels,  **kwargs)
        self.model_pre = self.model
        if self.kwargs['parallel'] :
            self.model = nn.DataParallel(self.model_pre, self.device_ids)
        else:
            self.model = self.model_pre

        self.model.cuda()
        self.model1.cuda()
        self.model2.cuda()
        self.model3.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.kwargs['optim']=='sgd' :
            self.optimizer = optim.SGD(parameters,lr = self.lr)
        elif self.kwargs['optim']=='adadelta' :
            self.optimizer = optim.Adadelta(parameters,lr = self.lr, weight_decay = \
                    self.kwargs['weight_decay'])
        elif self.kwargs['optim']=='adam' :
            self.optimizer = optim.Adam(parameters,lr = self.lr)
        elif self.kwargs['optim']=='sgd-mom' :
            self.optimizer = optim.SGD(parameters,lr = self.lr, momentum=0.9, nesterov =True)

    def loop_train(self, train_tuple, valid_tuple,test_tuple):

        filename = 'model_{}_useStructure_{}_fold_{}_nhids_{}_drate_{}_mark_{}_lam_{}'.format(\
                self.kwargs['data_path'].strip('/').split('/')[-1],not self.kwargs['no_structure'], self.kwargs['fold'], \
                self.kwargs['n_hids'],kwargs['drate'] ,kwargs['mark'],self.kwargs['lambda1'])
        self.save_folder = join(dirname(__file__), filename + time.strftime('-%b%d.%Hh%M', time.localtime()))

        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.logg = mylog(self.save_folder)
        self.logg.logging('Start training...')
        print('Start training...')

        with open(join(self.save_folder, 'ModelArgs.json'), 'w') as f:
            json.dump(self.kwargs, f)

        train_data = self.gen_batches(train_tuple)
        val_data = self.gen_batches(valid_tuple)
        test_data = self.gen_batches(test_tuple)

        N_nonupdate = 0
        self.batch_offset = 0
        try:
            for self.epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                update_flag = self.train(train_data,val_data)
                if not update_flag:
                    N_nonupdate += 1
                else:
                    N_nonupdate = 0
                if N_nonupdate > 50:
                    break

                 
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


        print('=' * 89)
        print('-'*30+'Testing Performance'+'-'*30)
        print('=' * 89)
        modelfilename = join(self.save_folder,"best-model.pkl")
        with open(modelfilename, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        test_loss, test_mse, test_reward, test_corr = self.validation(test_data)
        lenth = (-test_reward-test_mse)/self.kwargs['lambda1']
        self.logg.logging('| fold {}  | loss {:5.3f} | reward {:5.5} | MSE {:5.5}| Pearson\'s r {:5.5f} | average rationale length {:3.3f}'.format(args.fold, test_loss, test_reward,test_mse,test_corr, lenth))
        print('| fold {}  | loss {:5.3f} | reward {:5.5} | MSE {:5.5}| Pearson\'s r {:5.5f} | average rationale length {:3.3f}'.format(args.fold, test_loss, test_reward,test_mse,test_corr, lenth))
        return test_mse, test_corr

    def train(self,train_data, val_data):
        total_loss = 0
        total_mse = 0
        total_reward = 0
        start_time = time.time()
        n_batches = len(train_data[0])
        order = [i for i in xrange(n_batches)]
        random.shuffle(order)
        update_flag = False
        for i in range(n_batches):
            self.batch_offset += 1
            self.model.train()
            self.optimizer.zero_grad()


            if self.kwargs['no_structure']:
                data,nt,  targets = self.get_batch_nostructure(train_data, order[i], torch.FloatTensor)
                output = self.model(data,nt,None, targets)
            else:
                data,nt, fold, targets = self.get_batch(train_data, order[i], torch.FloatTensor)
                output = self.model(data,nt, fold, targets)

            batch_size = data.size()[0]
            reg_loss = torch.sum(output[0])
            # reg_loss = torch.mean(output[0])
            reg_loss.backward()

            ## update parameters
            if self.kwargs['clip_grad']>0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), kwargs['clip_grad'])

            self.optimizer.step()
            if self.kwargs['max_norm']>0 and (self.model_pre.fc.weight.norm() >self.kwargs['max_norm']).data.all():
               self.model_pre.fc.weight.data =( self.model_pre.fc.weight.data * self.kwargs['max_norm']) / self.model_pre.fc.weight.data.norm()

            reward =   torch.mean(output[1]).data[0]
            mse = torch.mean(output[2]).data[0]
            total_loss += reg_loss.data[0]
            total_mse += mse
            total_reward += reward

            if self.batch_offset % args.interval ==0:
                cur_loss = total_loss / args.interval
                cur_mse = total_mse / args.interval
                cur_reward = total_reward / args.interval
                elapsed = time.time() - start_time
                self.logg.logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.3f} | reward {:5.3} | MSE {:5.3f} |'.format(
                    self.epoch, i+1, n_batches, self.lr, elapsed * 1000 / args.interval,\
                            cur_loss, cur_reward, cur_mse))
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.3f} | reward {:5.3} | MSE {:5.3f} |'.format(
                    self.epoch, i+1, n_batches, self.lr, elapsed * 1000 / args.interval,\
                            cur_loss, cur_reward, cur_mse))

                total_loss = 0
                total_mse = 0
                total_reward = 0

                if self.batch_offset% (args.interval*4) ==0:
                    val_loss, mse, reward, corr = self.validation(val_data)
                    self.logg.logging('| epoch {:3d} | Validation |'\
                            'loss {:5.3f} | reward {:5.5} | MSE {:5.5} | corr {:5.5f} |'.format(
                                    self.epoch,val_loss, reward,\
                                            mse, corr))
                    print('| epoch {:3d} | Validation |'\
                            'loss {:5.3f} | reward {:5.5} | MSE {:5.5} | corr {:5.5f} |'.format(
                                    self.epoch,val_loss, reward,\
                                            mse, corr))
                    print('')

                    # test_loss, test_acc, test_reward, test_corr = self.validation(test_data)
                    # lenth = (-test_reward-test_acc)/self.kwargs['lambda1']
                    # self.logg.logging('| Testing  |'\
                    #         'loss {:5.3f} | reward {:5.5} | acc {:5.5}|corr {:5.5f} | len {:3.3f}'.\
                    #         format(test_loss, test_reward,test_acc,test_corr, lenth))
                    # testlen = (-test_reward-test_acc)/self.kwargs['lambda1']

                    vallen = (-reward-mse)/self.kwargs['lambda1']
                    try:
                        if np.isnan(corr):
                            value=0
                        else:
                            value = corr.tolist()
                    except:
                        pass
                    
                


                    # if not self.best_val_reward or (reward > self.best_val_reward and vallen<9 and testlen<9):
                    if not self.best_val_value or (value > self.best_val_value and vallen<9):
                        update_flag = True
                        filename = join(self.save_folder,
                                "best-model.pkl".format(corr))
                        with open(filename, 'wb') as f:
                            torch.save(self.model.state_dict(), f)
                            self.best_val_value = value
                        print('-'*15+'saved a new best model'+'-'*15)
                        self.logg.logging('-'*15+'saved a new best model'+'-'*15)

                start_time = time.time()
        return update_flag

    def validation(self, data_source, model=None):
        # Turn on evaluation mode which disables dropout.
        if model is None:
            model = self.model
        model.eval()
        total_loss = 0.0
        total_perf = 0
        mse = 0
        edge_mse = 0
        last_edge_mse = 0
        reward = 0
        n_batches = len(data_source[0])
        num_samples = 0
        preds = []
        labels = []


        for i in range(n_batches):
            if self.kwargs['no_structure']:
                data,nt,  targets = self.get_batch_nostructure(data_source, i, torch.FloatTensor, evaluation=True)
                output = model(data,nt, None, targets)
            else:
                data,nt, fold, targets = self.get_batch(data_source, i, torch.FloatTensor, evaluation=True)
                output = model(data,nt, fold, targets)

            batch_size = data.size()[0]
            mse += torch.mean(output[2]).data[0] * batch_size
            reward += torch.mean(output[1]).data[0] * batch_size
            total_loss += torch.sum(output[0]).data[0] * batch_size
            num_samples += batch_size

            pred = output[-1][0]
            pr = pred.data.cpu().numpy()
            preds.append(pr)
            labels.append(targets.view(-1,1).data.cpu().numpy())
        
        Y_pred = np.vstack(preds)
        Y = np.vstack(labels)

        # R^2 value for our predictions on the training set
        corr = scipy.stats.pearsonr(Y.flatten(),
                                   Y_pred.flatten())[0]

        return total_loss /num_samples,mse/num_samples, reward/num_samples, corr

    def cross_validation(self, data_source):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        total_loss = 0.0
        total_perf = 0
        acc = 0
        edge_acc = 0
        last_edge_acc = 0
        reward = 0
        n_batches = len(data_source[0])
        num_samples = 0
        preds = []
        labels = []
        
        for i in range(n_batches):
            print i
            data,nt, fold, targets = self.get_batch(data_source, i, torch.FloatTensor, evaluation=True)
            batch_size = data.size()[0]
            #output1 = self.model1(data,nt,fold, targets)
            #output2 = self.model2(data,nt,fold, targets)
            #output3 = self.model3(data,nt,fold, targets)

            output1 = self.model1(data,nt,None, targets)
            output2 = self.model2(data,nt,None, targets)
            output3 = self.model3(data,nt,None, targets)
 
            num_samples += batch_size

            pred1 = output1[-1][0]
            pred2 = output2[-1][0]
            pred3 = output3[-1][0]
            Y_pred1 = pred1.data.cpu().numpy()
            Y_pred2 = pred2.data.cpu().numpy()
            Y_pred3 = pred3.data.cpu().numpy()
            Y_pred = (Y_pred1+Y_pred2+Y_pred3)/3
            Y = targets.view(-1,1).data.cpu().numpy()
            preds.append(Y_pred)
            labels.append(Y)

        Y_pred = np.vstack(preds)
        Y = np.vstack(labels)
        mse = np.mean((Y-Y_pred)*(Y-Y_pred))
        corr = scipy.stats.pearsonr(Y.flatten(),Y_pred.flatten())[0]
        print('mse',mse, 'corr',corr)



    def evaluate(self, rawseq, word2id, leftflag=None):
        self.model.eval()
        seq = []
        ntseq = []
        n_seq = len(rawseq)/3
        for i in range(n_seq):
            seq.append(rawseq[3*i:3*i+3])
            ntseq += [x for x in rawseq[3*i:3*i+3]]
        
        window = int(self.kwargs['window'])
        padding =   window-len(seq)
        char_vocab = {'A':1,'C':2,'G':3, 'T':4}
        if leftflag is  None:
            seq = [word2id[x] for x in seq]
            ntseq = [char_vocab[x] for x in ntseq]
            maskids = None
        elif leftflag:
            seq =[0]*padding+ [word2id[x] for x in seq]
            ntseq = [0]*3*padding+ [char_vocab[x] for x in ntseq]
            maskids = [i for i in range(padding)]
        elif not leftflag:
            seq = [word2id[x] for x in seq] +[0]*padding
            ntseq =  [char_vocab[x] for x in ntseq]+[0]*3*padding
            maskids = [window-1 -i for i in range(padding)]
        assert len(seq)== window
        assert len(ntseq)== 3*window
        value = 0
        data = Variable(torch.LongTensor(seq)).view(1,-1).cuda()  
        ntdata = Variable(torch.LongTensor(ntseq)).view(1,-1).cuda()  
        target = Variable(torch.FloatTensor([value])).view(1,1).cuda()
        result = self.model(data, ntdata, None, target, maskids=maskids)

        pred = result[-1][0].data.cpu()[0]
        rationales = result[-1][1].data.cpu()[0].tolist()

        return pred, rationales, None , None

    def get_batch(self, source, i, warper_tensor = torch.LongTensor,evaluation=False, warp= True):
        data_ts  =  torch.LongTensor(source[0][i])
        nt_ts  =  torch.LongTensor(source[1][i])
        fold_ts  =  torch.FloatTensor(source[2][i])
        target_ts = warper_tensor(source[3][i])
        if True:
            data_ts = data_ts.cuda()
            nt_ts = nt_ts.cuda()
            fold_ts = fold_ts.cuda()
            target_ts = target_ts.cuda()
        if warp:
            data = Variable(data_ts, volatile=evaluation)
            nt = Variable(nt_ts, volatile=evaluation)
            fold = Variable(fold_ts, volatile=evaluation)
            target = Variable(target_ts)
            return data, nt,fold, target
        else:
            return data_ts,nt_ts, fold_ts, target_ts

    def get_batch_nostructure(self, source, i, warper_tensor = torch.LongTensor,evaluation=False, warp= True):
        data_ts  =  torch.LongTensor(source[0][i])
        nt_ts  =  torch.LongTensor(source[1][i])
        target_ts = warper_tensor(source[2][i])
        if True:
            data_ts = data_ts.cuda()
            nt_ts = nt_ts.cuda()
            target_ts = target_ts.cuda()
        if warp:
            data = Variable(data_ts, volatile=evaluation)
            nt = Variable(nt_ts, volatile=evaluation)
            target = Variable(target_ts)
            return data, nt, target
        else:
            return data_ts,nt_ts, target_ts

    def gen_batches(self,data_tuple):
        # suitable for abitrary length of data_tuple
        batches = [[] for i in xrange(len(data_tuple))]
        for i in xrange(len(data_tuple)-1):
            assert len(data_tuple[i]) == len(data_tuple[i+1])
        bs = self.args.batch_size
        for i in xrange(int(np.ceil(len(data_tuple[0]) / float(bs)))):
        # for i in xrange(int(np.floor(len(data_tuple[0]) / float(bs)))):  # delete the last batch
            for j in xrange(len(data_tuple)):
                batches[j].append(data_tuple[j][i * bs:i * bs + bs])
        return batches

    def gen_idx2word(self, vocab, id):
        idx2word = dict((word,idx) for idx,word in vocab.items())
        if idx2word.has_key(id):
            return idx2word[id]
        else:
            return 'unk'

    def testSeq(self,seq):
        window = int(self.kwargs['window'])
        n_seq  = len(seq)/3
        preds_sum = 0.0
        preds = [0]*n_seq
        position_imp = [0]*n_seq
        
        for i in range(n_seq):
            leftflag = None
            left = i-window/2
            right = i+window/2
            leftflag =None
            if left<0:
                left = 0
                leftflag = True
            if right>n_seq:
                right=n_seq
                leftflag = False
            testseq = seq[3*left:3*right]
            pred , ra, all_word, ra_words = self.evaluate( testseq, self.vocab, leftflag)
            preds[i] = pred
        return preds

if __name__ == "__main__":
    kwargs = {
        "n_repeat":3,
        "n_tokens":65,
        "device_ids":[0,1,2,3]

    }
        
    args = options.load_arguments()
    if args.fold is None:
        args.fold=0
        mse0,corr0 = main(args,kwargs)
        args.fold=1
        mse1,corr1 = main(args,kwargs)
        args.fold=2
        mse2,corr2 = main(args,kwargs)
        avg_mse = (mse0+mse1+mse2)/3
        avg_corr = (corr0+corr1+corr2)/3
        print('=' * 89)
        print('-'*30+'Overall Testing Performance'+'-'*30)
        print('=' * 89)
        print(' Cross-Validation| MSE {:5.5}| corr {:3.3f}'.\
                format(avg_mse, avg_corr))
    else:
        main(args,kwargs)




