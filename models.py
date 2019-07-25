import torch
import torch.nn as nn
import torch.nn.functional as F
import nntools.nntools as nntools
from torch.autograd import Variable
import nntools.utils as nnutils

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, cate_num, length, prate = 0.0, drate =0.0):
        super(PolicyNet, self).__init__()
        self.hidden_size = hidden_size
        self.cate_num = cate_num
        self.length = length+1
        self.prate = prate
        self.drate = drate

        self.dropout = nn.Dropout(self.drate)
        self.gen = nn.GRUCell(input_size+self.length, hidden_size)
        self.cf = nn.Linear(hidden_size, cate_num)

    def forward(self, inputs):
        '''
        inputs: (b_s,l,emb_size)
        return: (b_s,l)
        '''
        inputs_p = inputs
        b_s = inputs_p.size()[0]
        steps = inputs_p.size(1)
        outputs = Variable(torch.zeros(b_s,steps).cuda())
        pis = Variable(torch.zeros(b_s,steps).cuda())
        ones = Variable(torch.ones(b_s,1).cuda())
        probs = Variable(torch.zeros(b_s,steps).cuda())
        # prate is 0, indicating sampling from predicted policy distribution
        self.prate = 0.0 
        self.switch_m = Variable(torch.FloatTensor(b_s, 2).fill_(self.prate)).cuda()
        self.switch_m[:,1] = 1- self.prate
        self.action_m = Variable(torch.FloatTensor(b_s,self.cate_num).fill_(1.0/self.cate_num)).cuda()

        tag_onehot = Variable(torch.zeros(b_s,steps+1).cuda())
        h_t = self.init_hidden(b_s)
        for i in range(steps):
            input = inputs_p[:,i,:]
            tag = torch.sum(outputs,1, keepdim = True).long()
            tag_onehot.data.zero_()
            tag_onehot.scatter_(1,tag,ones)
            inp = torch.cat([input,tag_onehot],1) # (b_s, input+cate)
            h_t = self.gen(inp,  h_t)
            if self.drate>0:
                h_t = self.dropout(h_t)
            energe_s = nn.Softmax(1)(self.cf(h_t))

            if self.training:
                action_exploit = energe_s.multinomial()  # as edge_predict
                explorate_flag = self.switch_m.multinomial()  # as edge_predict
                action_explorate = self.action_m.multinomial()
                action = nntools.repackage_var(explorate_flag*action_exploit +\
                    (1-explorate_flag.float().float()).long() * action_explorate)
                # action = energe_s.multinomial()  # as edge_predict
            else:
                values,action = torch.max(energe_s,1)

            s_t = nntools.repackage_var(action.view(-1,1))
            pi = torch.gather(energe_s, 1, s_t) # (b_s,1), \pi for i,j
            pis[:,i] = pi
            probs[:,i:i+1] = energe_s[:,1:2]
            outputs[:,i] = s_t
        return  outputs, pis, probs

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(batch_size, self.hidden_size).zero_())

class RiboexpModel(nn.Module): 
    
    def __init__(self, n_labels = 2, **kwargs):
        super(RiboexpModel, self).__init__()

        seed = kwargs['seed']
        torch.manual_seed(seed)
        self.kwargs = kwargs
        self.drate = kwargs['drate']
        self.n_repeat = kwargs['n_repeat']
        self.n_hids = kwargs['n_hids']
        self.hidden_size = self.n_hids
        self.vocab_size = kwargs['n_tokens']
        self.length = int(kwargs['window'])
        self.L = kwargs['L']
        self.lambda1 =  kwargs['lambda1']

        # binary classification for policy net
        self.num_labels = n_labels

        self.input_n = 90
        self.foldsize = self.hidden_size/2
        self.encoder_g = nn.GRU(self.input_n, self.n_hids,1,dropout = self.drate,batch_first =True, bidirectional=True)
        self.generator = PolicyNet(2*self.n_hids, self.n_hids, self.num_labels,self.length,
                 drate = self.drate)
        self.encoder = nn.GRU(self.input_n, self.n_hids,1,dropout = self.drate, batch_first =True, bidirectional=True)
        
        if self.kwargs['no_structure']:
            self.n_fc = self.n_hids*2
        else:
            self.encoder_fold = nn.GRU(3+10, self.foldsize,1,dropout=self.drate, batch_first =True, bidirectional=True)
            self.n_fc = (self.n_hids+self.foldsize)*2

        self.drop = nn.Dropout(self.drate)
        self.fc = nn.Linear(self.n_fc,1)
        
    def forward(self, input, nt, fold, target, maskids=None):
        
        batch_size = input.size()[0]
        length = input.size()[1]

        x = nntools.onehot(input, self.vocab_size) # bs, len, 65
        xnt = nntools.onehot(nt, 5).view(batch_size,length,15) # bs, 10,15

        if maskids is not None:
            for id in maskids:
                x[:,id] = 0
                xnt[:,id] = 0

        if fold is not None:
            xfold = fold.view(batch_size, length,3)# bs, 10,3,

        if self.training:
            self.n_repeat = self.kwargs['n_repeat']
        else:
            self.n_repeat = 1

        positionx = Variable(torch.eye(length).repeat(batch_size,1,1)).cuda()
        x = torch.cat([x,xnt, positionx],2)

        inputs_r = x.repeat(self.n_repeat,1,1) # (b_s*5,l,emb_size)
        targets_r = target.view(-1,1).repeat(self.n_repeat,1)

        h0 = Variable(torch.zeros( 2,batch_size*self.n_repeat, self.hidden_size)).cuda()
        g_features,hidden0 = self.encoder_g(inputs_r, h0)
        tag_outputs, pis, probs = self.generator(g_features)

        if maskids is not None:
            for id in maskids:
                tag_outputs[:,id] = int(0)


        mask_words = tag_outputs.unsqueeze(2).repeat(1,1,self.input_n) * inputs_r #bs,len,emb
        forward_features,hidden0 = self.encoder(mask_words, h0)

        
        if fold is not None:
            h0 = Variable(torch.zeros( 2,batch_size*self.n_repeat, self.foldsize)).cuda()
            xfold = torch.cat([xfold, positionx],2).repeat(self.n_repeat,1,1)
            forward_features_fold,hidden0 = self.encoder_fold(xfold, h0)
            forward_features = torch.cat([forward_features, forward_features_fold],2)

        rnn_o = nn.MaxPool1d(forward_features.size(1))(forward_features.permute(0,2,1))
        rnn_o = rnn_o.squeeze()


        fc1_feed = self.drop(rnn_o)
        logit = self.fc(fc1_feed)
        l2 =  (logit - targets_r)*(logit - targets_r)
        distance = l2

        '''supervised learning'''
        self.squareloss = torch.mean(l2)# /self.n_repeat
        # '''30-5.47'''
        R_l2 =  -self.lambda1 * nn.ReLU()(torch.norm(tag_outputs,1,1,keepdim = True)-self.L)
        reward = -distance + R_l2
        avg_reward = torch.mean(reward.view(-1,batch_size),0).repeat(self.n_repeat,1).view(-1,1)
        real_reward = nn.ReLU()(reward - avg_reward)
        real_reward = nntools.repackage_var(real_reward)
        rlloss = -torch.sum( torch.log(pis.clamp(1e-6,1)),1,keepdim=True)*real_reward
        self.rlloss = torch.mean(rlloss)# /self.n_repeat
        eta = 0.5
        self.loss = eta*l2 + (1-eta)*rlloss

        self.accuracy = torch.mean(l2)
        self.reward = torch.mean(reward)

        rationales = tag_outputs

        return  self.loss, reward, self.accuracy, (logit, rationales, probs)







