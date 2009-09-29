#! /usr/bin/env python

import numpy
import collections
from math import log, fabs
from plsa import *
from corpus import *

    

class MAPTopicInference(TopicInference):
    """MAP Topic Inference"""
    
    def __init__(self, numTopics, corpus, alpha=1.1, eta=1.1):
        super(MAPTopicInference, self).__init__(numTopics, corpus)
        self.alpha_minus_1 = numpy.ones(numTopics) * (alpha - 1)
        self.eta_minus_1   = numpy.ones(numTopics) * (eta - 1)

    def calc_updated_gamma_w_j(self, w, j):
        """calculate updated gamma values for a given word and doc index"""
        # g = self.phi[w] * self.theta[j]
        g = (self.n_wk[w] + self.eta_minus_1) * (self.n_kj[j] + self.alpha_minus_1) / (self.n_k + self.numTypes * self.eta_minus_1)
        g /= g.sum()
        return g

    def calc_updated_phi_w(self, w):
        """calc the updated phi value for a given word index"""
        p = (self.n_wk[w] + self.eta_minus_1)/(self.n_k + self.numTypes * self.eta_minus_1)
        p /= p.sum()
        return p

    def calc_updated_theta_j(self, j):
        """calc the updated theta value for a given doc index"""
        return (self.n_kj[j] + self.alpha_minus_1)/(self.n_j[j] + self.numTopics * self.alpha_minus_1)
        

         
def main(argv=None):
    if argv is None:
        argv = sys.argv

    doc_directory = ""
    numTopics     = 10

    if len(argv) > 1:
        doc_directory = argv[1]

    if len(argv) > 2:
        numTopics = int(argv[2])

    corpus = Corpus()
    corpus.readfiles(doc_directory)
    #print corpus.vocabulary

    topics = MAPTopicInference(numTopics, corpus)
    
    if topics.iterate(100, 0.000001) < 0.000001:
        topics.printPhi()
        topics.printTheta()
        
    return topics
    

if __name__ == '__main__':
    sys.exit(main())
