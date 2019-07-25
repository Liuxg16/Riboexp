#!/usr/bin/python3
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import functional, init
from torch.nn import Parameter

def repackage_var(vs, requires_grad = False):

    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(vs) == Variable:
        return Variable(vs.data, requires_grad = requires_grad)
    elif type(vs) == Parameter:
        return Parameter(vs.data,requires_grad = requires_grad)
    else:
        return tuple(repackage_var(v) for v in vs)

def onehot(data1, n_dimension):
    n_dim = data1.dim()
    batch_size = data1.size()[0]
    data = data1.view(-1,1)  
    if hasattr(data1,'data'):
        assert  (torch.max(data1)< n_dimension).data.all() # bs,1
        y_onehot = Variable(torch.FloatTensor(data.size(0),n_dimension).zero_())
        ones = Variable(torch.FloatTensor(data.size()).fill_(1))
    else:
        y_onehot = torch.FloatTensor(data.size(0),n_dimension).zero_()
        ones = torch.FloatTensor(data.size()).fill_(1)

    if data.is_cuda:
        y_onehot = y_onehot.cuda()
        ones = ones.cuda()

    y_onehot.scatter_(1,data,ones)
    if n_dim ==1:
        return y_onehot.view(batch_size,n_dimension)
    elif n_dim ==2:
        return y_onehot.view(batch_size,-1,n_dimension)

def cal_loss(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_()).cuda()
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1)).cuda()
    y_onehot.scatter_(1,target_label,ones)
    log_dis = torch.log(distribution) 
    loss = torch.sum(-y_onehot*log_dis, dim = 1)
    return loss.view(-1,1) #(b_s,1)
 
def cal_loss_cpu(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_())
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1))
    # print target_label
    # print ones
    # print y_onehot
    y_onehot.scatter_(1,target_label,ones)
    log_dis = torch.log(distribution) 
    loss = torch.sum(-y_onehot*log_dis, dim = -1)
    return loss #(b_s,1)

def cal_sf_loss(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_()).cuda()
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1)).cuda()
    y_onehot.scatter_(1,target_label,ones)
    dec_out = nn.LogSoftmax()(distribution) 
    loss = torch.sum(-y_onehot*dec_out, dim = -1)
    return loss
 
def map_tensor(map, tensor):
    shape = tensor.size()
    data = tensor.view(-1).tolist()
    # print data
    data_map = [map[i] for i in data]
    return torch.FloatTensor(data_map).view(shape)

class RNN_lxg(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_in, nhid, nlayers, dropout):
        super(RNN_lxg, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_in, nhid, int(nlayers), dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
 #       self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = int(nhid)
        self.nlayers = int(nlayers)

    def forward(self, inputs, inputs_mask = None, hidden = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        if inputs_mask == None:
            inputs_mask_p = Variable(torch.ones(inputs_p.size()[:-1]).cuda())
        else:
            inputs_mask_p = inputs_mask.permute(1,0).contiguous() 
        steps = len(inputs_p)
        b_s = inputs_p.size()[1]
        
        if hidden == None:
            hidden = self.init_hidden(b_s)

        outputs = Variable(torch.zeros(steps, b_s, self.nhid).cuda())
        for i in range(steps):
            input = inputs_p[i]
            input_mask = inputs_mask_p[i]
            output, hidden = self.step(input,input_mask, hidden)
            outputs[i] = output
        return outputs

    def step(self, input ,input_mask, hidden):
        input_r = input.unsqueeze(0) #(1,shape)
        output_p, hidden1 = self.rnn(input_r, hidden)
        mask =  input_mask.view(1,-1,1).expand_as(output_p)
        output_p = output_p *mask + hidden[0].unsqueeze(0)* (1-mask)
        return output_p, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_om = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)

        self.init_weights()

    def forward(self, inputs, input_m = None, hidden = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        if hidden is  not None:
            h_t,c_t = hidden
        else:
            h_t,c_t = self.init_hidden(b_s)

        if input_m is not None:
            inputs_mask_p = input_m.permute(1,0).contiguous() 
        else:
            inputs_mask_p = Variable(torch.ones(inputs_p.size()[:-1]).cuda())
        steps = len(inputs_p)

        outputs = Variable(torch.zeros(b_s,steps, self.hidden_size).cuda())

        for i in range(steps):
            input = inputs_p[i]
            input_mask = inputs_mask_p[i]
            h_t, c_t = self.step(input, input_mask, h_t, c_t)
            outputs[:,i,:] = h_t

  #       result = outputs.permute(1,0,2).contiguous() 
        return  outputs,(h_t,c_t)

    def step(self, inp, input_mask, h_0, c_0):
        # forget gate
        f_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        # f_g = F.sigmoid(self.weight_fx(inp) + self.weight_fm(h_0))
        # input gate
        i_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))
        # output gate
        o_g = nn.Sigmoid()(self.weight_ox(inp) + self.weight_om(h_0))
        # intermediate cell state
        c_tilda = nn.Tanh()(self.weight_cx(inp) + self.weight_cm(h_0))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # hidden state
        hx = o_g * nn.Tanh()(cx)

        mask =  input_mask.view(-1,1).expand_as(hx)
        ho = hx *mask + h_0 * (1-mask) # (1,b_s,hids)
        return ho, cx

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()),
                Variable(weight.new(batch_size, self.hidden_size).zero_()))


    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_om.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)
        self.weight_ox.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_om.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)
        self.weight_ox.bias.data.fill_(0)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        # self.init_weights()

    def forward(self, inputs, input_m = None, hidden = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        if hidden is  not None:
            h_t = hidden
        else:
            h_t = self.init_hidden(b_s)
        steps = len(inputs_p)
        hts = []
        for i in range(steps):
            input = inputs_p[i]
            h_t= self.step(input, h_t)
            hts.append(h_t.unsqueeze(0))
        outputs = torch.cat(hts,0)
        return  outputs.permute(1,0,2),h_t

    def step(self, inp, h_0):
        # forget gate
        z_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        r_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))
        h_tilda = nn.Tanh()(self.weight_cx(inp) + r_g * self.weight_cm(h_0))
        # current cell state
        h_t = (1-z_g) * h_0 + z_g * h_tilda

        # mask =  input_mask.view(-1,1).expand_as(h_t)
        # ho = h_t *mask + h_0 * (1-mask) # (1,b_s,hids)
        return h_t

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()))

    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)

class GRU_F(nn.Module):
    '''
    add feedback
    '''
    def __init__(self, input_size, hidden_size, cate_num):
        super(GRU_F, self).__init__()
        self.hidden_size = hidden_size
        self.cate_num = cate_num
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size+self.cate_num, hidden_size)
        self.weight_ix = nn.Linear(input_size+self.cate_num, hidden_size)
        self.weight_cx = nn.Linear(input_size+self.cate_num, hidden_size)
        self.cf = nn.Linear(hidden_size, cate_num)

    def forward(self, inputs,targets, input_m = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        h_t = self.init_hidden(b_s)

        if input_m is not None:
            inputs_mask_p = input_m.permute(1,0).contiguous() 
        else:
            inputs_mask_p = Variable(torch.ones(inputs_p.size()[:-1]).cuda())

        steps = len(inputs_p)
        outputs = Variable(torch.zeros(b_s,steps, self.cate_num).cuda())
        ones = Variable(torch.ones(b_s,1).cuda())
        cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
        for i in range(steps):
            input = inputs_p[i]
            input_mask = inputs_mask_p[i]
            h_t= self.step(input, input_mask, h_t, cate_tm1)
            output = nn.Softmax()(self.cf(h_t))
            outputs[:,i,:] = output
            if  self.training:
                cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
                cate_tm1.scatter_(1,targets[:,i].contiguous().view(-1,1),ones)
            else:
                value,pred_t1 = torch.max(output,1)
                cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
                cate_tm1.scatter_(1,pred_t1.view(-1,1),ones)

        return  outputs

    def step(self, input_x, input_mask, h_0, cate_tm1):

        inp = torch.cat([input_x,cate_tm1],1) # (b_s, input+cate)
        # forget gate
        z_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        # input gate
        r_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))

        h_tilda = nn.Tanh()(self.weight_cx(inp) + r_g * self.weight_cm(h_0))
        # current cell state
        h_t = (1-z_g) * h_0 + z_g * h_tilda

        mask =  input_mask.view(-1,1).expand_as(h_t)
        ho = h_t *mask + h_0 * (1-mask) # (1,b_s,hids)
        return ho

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()))

    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)

class  Fast_Text(nn.Module):
    
    def __init__(self, size_filter, n_out_kernel, embsize, drate = 0.01):
        super(Fast_Text,self).__init__()
        self.size_filter = size_filter 
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        # Ks = list(range(1,size_filter, int(size_filter/5)))  # [1, 2, 3, 4, 5] # args.kernel_sizes
        Ks = [1,2,3,4,5]
        self.n_out = len(Ks) * self.n_filter
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3

    def forward_locate(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        v, idxs = torch.max(x1[4],2)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3,idxs

class  SentCNN(nn.Module):
    
    def __init__(self, size_filter, n_out_kernel, embsize, drate = 0.01):
        super(SentCNN,self).__init__()
        self.size_filter = size_filter 
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        Ks = size_filter
        self.n_out = len(Ks) * self.n_filter
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3

    def forward_locate(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        v, idxs = torch.max(x1[4],2)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3,idxs

class  Fix_CNN(nn.Module):
    '''
    give a summarization about the text
    (bs,n,emb)->(bs,k)
    '''
    
    def __init__(self, filter_list, n_out_kernel, embsize, drate = 0.01):
        super(Fix_CNN,self).__init__()
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        Ks = filter_list
        self.n_out = len(Ks) * self.n_filter
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        # self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3

    def forward_locate(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        v, idxs = torch.max(x1[4],2)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3,idxs

class  CNNEncoder(nn.Module):
    
    def __init__(self, Kernel, n_out_kernel, embsize, drate = 0.01):
        super(CNNEncoder,self).__init__()
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        K = Kernel # size_filter
        self.convs1 = nn.Conv2d(Ci, Co, (K, embsize),padding = (K-1)/2) 
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = F.relu(self.convs1(x))  #[(N,Co,len), ...]*len(Ks)
        x2 = nn.MaxPool2d((1,x1.size(3)),(1,1))(x1).squeeze(3) 
        return x2

class  CNN(nn.Module):
    
    def __init__(self, len_filter, n_out_kernel, embsize):
        super(CNN,self).__init__()
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.conv = nn.Conv2d(Ci, Co, (len_filter, embsize),padding = (int((len_filter)/2),0)) 
        self.conv1d = nn.Conv1d(Co, 2*Co, len_filter, padding = (int((len_filter)/2),0))
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.poolsize = 3

    def conv_and_pool(self, x, conv):
        '''
        x: (b_s, 1, W, l)
        return: (b_s,co,(W-len_filter+1)/poolsize) 
        '''
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, self.poolsize)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x = self.conv_and_pool(x,self.conv)
        x = self.conv1d(x)
        x = F.max_pool1d(x, x.size()[2]).squeeze(2)
        return x

class CNN_Text(nn.Module):

    def __init__(self, size_filter, n_out_kernel, embsize, drate=0.05):
        """
        - size_filter: like [1, 2, 3, 4, 5]
        - n_out_kernel: like 10
        """
        super(CNN_Text,self).__init__()
        self.size_filter = size_filter
        assert(size_filter > 4)
        self.drate = drate
        self.embsize = embsize
        Co = n_out_kernel
        Ci = 1 #n_in_kernel
        size_filter += 1 
        # poch = int(round(size_filter/5.0)) if (size_filter/5.0)>1 else 1
        Ks = [3,4,5]# list(range(1,size_filter, poch))
        self.n_out = len(Ks) * n_out_kernel
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        length = x.size(1)
        x = x.unsqueeze(1) # (N,Ci,len,embsize), N = bs, Ci = 1
        x = [F.relu(conv(x)).squeeze(3)[:,:,:length] for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)

        x = torch.cat(x, 1) # (N, co*len(Ks), len)
        x = x.permute(2, 0, 1)  # (length, batch, len(KS)*Co = embsize)
        return x

class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FulyConnect(nn.Module):

    def __init__(self, n_in, n_out):
        super(FulyConnect, self).__init__()
        self.W = nn.Parameter(torch.rand(n_in, n_out))
        self.b = nn.Parameter(torch.rand(1, n_out))

    def forward(self, x, gamma = 1.0):
        if self.training:
            return torch.mm(x,self.W)+self.b
        else:
            return torch.mm(gamma*x,self.W)+ self.b





def compute_performance(label, pred_label):
    """
    1: start
    2: middle
    3: end
    """
    pcs = 0
    recall = 0
    f1 = 0

    def gen_entities(l):
        del_ixs = []
        entities = dict()
        ixs,_ = zip(*filter(lambda x: x[1] == 1, enumerate(l)))
        ixs = list(ixs)
        ixs.append(len(label))
        for i in range(len(ixs) -1):
            sub_label = l[ixs[i]:ixs[i + 1]]
            end_mark = max(sub_label)
            end_ix = ixs[i] + sub_label.index(end_mark) + 1
            entities["{}_{}".format(ixs[i], end_ix)] = l[ixs[i]:end_ix]
            del_ixs.extend(range(ixs[i], end_ix))
        return entities, del_ixs

    # g_entities, _ = gen_entities(label)
    # p_entities, del_ixs = gen_entities(pred_label)

    try:
        g_entities, _ = gen_entities(label)
        p_entities, del_ixs = gen_entities(pred_label)
        # print(g_entities)
        # print(p_entities)
        label_span = g_entities.keys()
        pred_span = p_entities.keys()

        A_span = label_span  & pred_span

        A = len(A_span)
        B = len(pred_span)
        C = len(label_span)
      
        if B == 0: B = 1e-6
        if C == 0: C = 1e-6


        #----------------------------------------------
        #ERROR: divide by 0 
        # pcs = 0.5
        # recall = 0.5
        pcs = A / float(B)
        recall = A / float(C)
        # print("---------------------------- Print pcs and recall -----------------------------")
        # print(" PCS -> ", pcs)
        # print(" Recall -> ", recall)
        
        #f1 = 1
        f1 = 2 * pcs * recall / float(pcs + recall)
        return pcs, recall, f1
    
    except:
        return 0.0,0.0,0.0

def compute_performance1(label, pred_label):
    """
    1: start
    2: middle
    3: end
    """
    pcs = 0
    recall = 0
    f1 = 0

    def gen_entities(l):
        del_ixs = []
        entities = dict()
        ixs,_ = zip(*filter(lambda x: x[1] == 1, enumerate(l)))
        ixs = list(ixs)
        ixs.append(len(label))
        for i in range(len(ixs) -1):
            sub_label = l[ixs[i]:ixs[i + 1]]
            end_mark = max(sub_label)
            end_ix = ixs[i] + sub_label.index(end_mark) + 1
            entities["{}_{}".format(ixs[i], end_ix)] = l[ixs[i]:end_ix]
            del_ixs.extend(range(ixs[i], end_ix))
        return entities, del_ixs

    # g_entities, _ = gen_entities(label)
    # p_entities, del_ixs = gen_entities(pred_label)

    try:
        g_entities, _ = gen_entities(label)
        p_entities, del_ixs = gen_entities(pred_label)
        print(g_entities)
        print(p_entities)
        count_tp = 0
        count_fp = 0

        for p_t in p_entities.keys():
            try:
                if p_entities[p_t] == g_entities[p_t]:
                    count_tp += 1
                else:
                    count_fp += 1
            except:
                count_fp += 1
        temp_labels = zip(label, pred_label, range(len(label)))
        temp_labels = filter(lambda x: x[2] not in del_ixs, temp_labels)
        label, pred_label, _ = zip(*temp_labels)

        count_tn = 0
        count_fn = 0

        for pred_l, l in zip(label, pred_label):
            if pred_l != 0:
                count_fp += 1
            elif l == 0:
                count_tn += 1
            else:
                count_fn += 1
      
        if count_tn == 0: count_tn = 1
        if count_fn == 0: count_fn = 1
        if count_tp == 0: count_tp = 1
        if count_fp == 0: count_fp = 1


        #----------------------------------------------
        #ERROR: divide by 0 
        # pcs = 0.5
        # recall = 0.5
        pcs = count_tp / float(count_tp + count_fp)
        recall = count_tp / float(count_tp + count_fn)
        # print("---------------------------- Print pcs and recall -----------------------------")
        # print(" PCS -> ", pcs)
        # print(" Recall -> ", recall)
        
        #f1 = 1
        f1 = 2 * pcs * recall / float(pcs + recall)
        return pcs, recall, f1
    
    except:
        pass 

def cal_performance(pred, label, mask, label_map):
    '''
    for conll
    '''
    preds = pred.data.cpu().tolist()        
    labels = label.data.cpu().tolist()
    masks = mask.data.cpu().tolist()
    A,B,C = 0,0,0
    for (p,l),m in zip(zip(preds,labels),masks):
        num = int(sum(m))
        chunks_p = utils.iob_to_spans(p[:num],label_map)
        chunks_l = utils.iob_to_spans(l[:num],label_map)
        A += len(chunks_p & chunks_l)
        B += len(chunks_p)
        C += len(chunks_l)
    return A,B,C





