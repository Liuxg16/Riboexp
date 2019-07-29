
import argparse
def load_arguments():

    parser = argparse.ArgumentParser(description='Hyperparameters of Riboexp')
    parser.add_argument('--data_path', type= str, default=None,
            help='the dir of the dataset')
    parser.add_argument('--gene_path', type= str, default=None,
            help='the file path for the prediction of a particular gene')
    parser.add_argument('--no_structure', action='store_true',
            help='flag indicating not using the rna structure information')
    parser.add_argument('--n_hids', type=int, default=512,
            help='number of hidden units per layer: {256,360,512,640}')
    parser.add_argument('--lambda1', type=float, default=0.0083,
            help='penalty factor of sparsity, {7.5-8.5}*0.001')
    parser.add_argument('--lr', type=float, default= 1,
            help='initial learning rate: {0.01, 0.1, 0.5,1}')
    parser.add_argument('--clip_grad', type=float, default=1,
            help='gradient clipping: {0.25,0.5, 0.1, 1, 2,5}')
    parser.add_argument('--epochs', type=int, default=200,
            help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=600, metavar='N',
            help='batch size: 1-1000')
    parser.add_argument('--drate', type=float, default= 0.4,
            help='dropout applied to layers: 0-1 (0 = no dropout)')
    parser.add_argument('--max_norm', type=float, default= 3,
            help='dropout applied to layers:0-10 (0 = not use it)')
    parser.add_argument('--weight_decay', type=float, default= 0.0,
            help='weight regularization:0-1')
    parser.add_argument('--load', type = str, default = None,
                        help='file path to load model')
    parser.add_argument('--load0', type = str, default = None,
                        help='file path to load model1 for cv test')
    parser.add_argument('--load1', type = str, default = None,
                        help='file path to load model2 for cv test')
    parser.add_argument('--load2', type = str, default = None,
                        help='file path to load model3 for cv test')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--parallel', action='store_true',
                        help='use multi gpus')
    parser.add_argument('--interval', type=int, default=20, metavar='N',
                        help='report interval, 0-100')
    parser.add_argument('--mark', type=str,  default='mark',
                        help='note to highlignt')
    parser.add_argument('--optim', type=str,  default='adadelta',
            help='the name of optimizer:adadelta, sgd')
    parser.add_argument('--mode', type=int, default=0,
            help='model mode, 0:train,1:eval,2:cv')
    parser.add_argument('--window', type=float, default=10,
            help='the length of a fragment:6-12')
    parser.add_argument('--L', type=float, default=0,
            help='maximum length of penalty free: 0-10')
    parser.add_argument('--fold', type=int, default=None,
            help='fold index for cross-validation:0,1,2')
    args = parser.parse_args()

    return args

