#! /usr/bin/env python

from __future__ import with_statement

from corpus import *
# import os.path
# import re
# import sys
# import collections
import numpy




class TopicInference(object):
    """docstring for TopicInference"""
    
    def __init__(self, numTopics, corpus):
        super(TopicInference, self).__init__()
        self.numTopics = numTopics
        self.corpus = corpus

        self.numDocs  = len(corpus)
        self.numTypes = len(corpus.vocabulary)

        #self.gamma = numpy.ones((self.numDocs, self.numTypes, self.numTopics))
        self.gamma = numpy.random.random((self.numDocs, self.numTypes, self.numTopics))
        # self.phi   = numpy.zeros((self.numTypes, self.numTopics))
        # self.theta = numpy.zeros((self.numTopics, self.numDocs))

        for d in self.gamma:
            for w in d:
                w /= w.sum()

        def countArray(doc):
            return numpy.array([doc.counts[w] for w in corpus.vocabulary])
        
        self.observations = numpy.array([countArray(doc) for doc in corpus])

        
    def em(self):
        """docstring for expectation"""
        n_wj      = self.observations.transpose()
        gamma_wjk = self.gamma.transpose((1,0,2))
        
        n_wk = numpy.array([numpy.dot(n_wj[w], gamma_wjk[w]) for w in xrange(self.numTypes)])
        n_k  = n_wk.sum(axis=0)
        
        self.phi = n_wk/n_k
        for w in self.phi:
            w /= w.sum()
        
        n_jw      = self.observations
        gamma_jwk = self.gamma
        
        n_kj = numpy.array([numpy.dot(n_jw[j], gamma_jwk[j]) for j in xrange(self.numDocs)])
        n_j  = n_kj.sum(axis=0)
        
        self.theta = n_kj/n_j
        for j in self.theta:
            j /= j.sum()

        for j in xrange(self.numDocs):
            self.gamma[j] = n_wk * n_kj[j] / n_k
            for w in self.gamma[j]:
                w /= w.sum()
        
    def printPhi(self):
        """docstring for printPhi"""
        for w in xrange(self.numTypes):
            print "%20s" % self.corpus.vocabulary[w],
            for k in xrange(self.numTopics):
                print "%6.4f" % self.phi[(w,k)],
            print

    def printTheta(self):
        """docstring for printTheta"""
        for j in xrange(self.numDocs):
            print "%20s" % self.corpus[j].name(),
            for k in xrange(self.numTopics):
                print "%6.4f" % self.theta[(j,k)],
            print

    def iterate(self):
        """docstring for iterate"""
        self.em()
        self.printPhi()
        self.printTheta()
        

        
    def __str__(self):
        """show topic assignments"""
        return self.gamma.__str__() 
    
                
         
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
    print corpus.vocabulary

    topics = TopicInference(numTopics, corpus)
    
    print topics.observations
    print topics.gamma
    
    topics.iterate()
    topics.iterate()
    topics.iterate()
    
    return topics

if __name__ == '__main__':
    sys.exit(main())
