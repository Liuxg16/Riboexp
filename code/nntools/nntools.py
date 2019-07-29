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





