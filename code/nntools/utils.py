try: # python2
    import cPickle
    from itertools import izip
except ImportError:
    import _pickle as cPickle

# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.use('Agg')
import time, os
import re
import torch
import numpy as np

import logging,sys
logger = logging.getLogger(__name__)


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

class MyRecorder(object):
    """
    MyRecorder keeps a dict of things, every entry in the dict is a list.
    """

    def __init__(self, savefolder=None):
        self.content = {}
        self.savefolder = savefolder

    def add(self, content_dict):
        """
        content_dict should be a dict, key is the entry name; value is the entry content.
        """
        '''for python3'''
        if sys.version > '3':
            for key,value in content_dict.items():
                if key not in self.content:
                    self.content[key] = [value]
                else:
                    self.content[key].append(value)
        else:
            for key,value in content_dict.iteritems():
                if not self.content.has_key(key):
                    self.content[key] = [value]
                else:
                    self.content[key].append(value)

    def save(self):
        with open(os.path.join(self.savefolder, "records.pkl"), 'w') as f:
            cPickle.dump(self.savefolder, f, -1)

    def visualize(self, IFSHOW=False):
        for key, value in self.content.items():
            if isinstance(value[0], int) or isinstance(value[0], float):
                plt.figure(key)
                plt.title(key)
                series = numpy.array(value)
                plt.plot(series, label=key)
                plt.legend(loc='best')
                if self.savefolder:
                    plt.savefig(os.path.join(self.savefolder, key))
            elif isinstance(value[0], numpy.float):
                pass
            else:
                print("type wrong")
            if IFSHOW:
                plt.show()

class mylog(object):

    def __init__(self,dirname, level=logging.DEBUG):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        filename = dirname+'/logging'+time.strftime('-%b%d.%Hh%M', time.localtime())+'.log'
        logname = time.strftime('-%b%d.%Hh%M', time.localtime())
    	handler = logging.FileHandler(filename)        
    	handler.setFormatter(formatter)

    	logger = logging.getLogger(logname)
    	logger.setLevel(level)
    	logger.addHandler(handler)
    	self.loggg = logger

    def logging(self,str):
        self.loggg.info("\n"+str)




if __name__ == "__main__":
    pass

