#! /usr/bin/env python

from __future__ import with_statement

from corpus import *
# import os.path
# import re
# import sys
# import collections
import numpy
import collections
from math import log, fabs


detailed_trace = False


class DocumentTopicAssignment(collections.defaultdict):
    """docstring for DocumentTopicAssignment"""
    
    def __init__(self, numTopics, doc):

        def random_vector():
            """initialize a random vector"""
            vec = numpy.random.random((numTopics))
            vec /- vec.sum()
            return vec

        def zero_vector():
            """initialize a zero vector"""
            return numpy.zeros((numTopics))

        super(DocumentTopicAssignment, self).__init__(zero_vector)
        self.doc = doc
        # self.gamma_j = collections.defaultdict(random_vector)


    # def __getitem__(self, key):
    #     """delegate to gamma_j"""
    #     return self.gamma_j[key]
    #     
    # def __contains__(self, item):
    #     """delegate to gamma_j"""
    #     return item in self.gamma_j
    
    

class TopicInference(object):
    """docstring for TopicInference"""
    
    def __init__(self, numTopics, corpus):
        super(TopicInference, self).__init__()
        self.numTopics = numTopics
        self.corpus = corpus

        self.numDocs  = len(corpus)
        self.numTypes = len(corpus.vocabulary)
        
        self.gamma = [DocumentTopicAssignment(self.numTopics, doc) for doc in self.corpus]    # posterior
            # gamma is (j,w,k)
            
        self.phi   = numpy.zeros((self.numTypes, self.numTopics))
        self.theta = numpy.zeros((self.numDocs,  self.numTopics))
        
        # for p in self.phi:
        #     p /= p.sum()
        # 
        # for p in self.theta:
        #     p /= p.sum()

        self.n_wk   = numpy.random.random((self.numTypes, self.numTopics))
        self.n_kj   = numpy.random.random((self.numDocs, self.numTopics))

        self.n_k    = self.n_wk.sum(axis=0)
        self.n_j    = self.n_kj.sum(axis=1)


    def expectation(self):
        """update gamma using new param values"""
        for j in xrange(self.numDocs):
            for w in xrange(self.numTypes):
                word = self.corpus.vocabulary[w]
                # g = self.phi[w] * self.theta[j]
                g = self.n_wk[w] * self.n_kj[j] / self.n_k
                g /= g.sum()
                self.gamma[j][word] = g
                
    def maximization(self):
        """calculate MLE for phi and theta values"""
        
        self.n_wk *= 0
        self.n_kj *= 0

        self.n_k  *= 0
        self.n_j  *= 0
        
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            for j in xrange(self.numDocs):
                p_topic = self.gamma[j][word] * self.corpus[j][word]
                
                if detailed_trace:
                    print "p_topic %3d %20s %s" % (j, word, p_topic)
                    
                self.n_wk[w] += p_topic
                self.n_kj[j] += p_topic
                # self.n_k     += p_topic

        self.n_k    = self.n_wk.sum(axis=0)
        self.n_j    = self.n_kj.sum(axis=1)
                

        if detailed_trace:
            print "n_wk:"
            for w in xrange(self.numTypes):
                word = self.corpus.vocabulary[w]
                print "\t%20s %s" % (word, self.n_wk[w])
            
            print "n_kj:"
            for j in xrange(self.numDocs):
                print "\t%20s %s" % (j, self.n_kj[j])
            
            print "n_k:\t", self.n_k
        
        for w in xrange(self.numTypes):
            p = self.n_wk[w]/self.n_k
            p /= p.sum()
            self.phi[w] = p

        for j in xrange(self.numDocs):
            #self.n_j = self.n_kj[j].sum()
            self.theta[j] = self.n_kj[j]/self.n_j[j]


    def logLikelihood(self):
        """docstring for logLikelihood"""
        l = numpy.dot(self.phi, self.theta.transpose())
        ll = 0.0
        for j in xrange(self.numDocs):
            for w in xrange(self.numTypes):
                word = self.corpus.vocabulary[w]
                n = self.corpus[j][word]
                if n > 0:
                    p = l[(w,j)]
                    print "p %d %20s %f %d" % (j, word, p, n)
                    ll += n*log(p)
        return ll
        
        
    def em(self):
        """docstring for em"""
        
        self.expectation()
        self.maximization()

        
    def printPhi(self):
        """docstring for printPhi"""
        print "phi:"
        for w in xrange(self.numTypes):
            print "%20s" % self.corpus.vocabulary[w],
            for k in xrange(self.numTopics):
                print "%6.4f" % self.phi[(w,k)],
            print

    def printTheta(self):
        """docstring for printTheta"""
        print "theta:"
        for j in xrange(self.numDocs):
            print "%20s" % self.corpus[j].name(),
            for k in xrange(self.numTopics):
                print "%6.4f" % self.theta[(j,k)],
            print

    def printGamma(self):
        print "gamma:"
        for j in xrange(self.numDocs):
            print "    %s:" % self.corpus[j].name()
            for word in self.corpus.vocabulary:
                print "%20s" % word,
                for k in xrange(self.numTopics):
                    print "%6.4f" % self.gamma[j][word][k],
                print

    def printAll(self):
        """docstring for printAll"""
        self.printPhi()
        self.printTheta()
        # if detailed_trace:
        self.printGamma()


    def iterate(self, numIterations=1, convergence=0.01, startIteration=1):
        """docstring for iterate"""
        
        prev_ll = 0
        for i in xrange(numIterations):
            print "iteration %d: " % (i+startIteration)
            self.em()
            ll = self.logLikelihood()
            diff = fabs((ll - prev_ll)/ll)
            prev_ll = ll
            print "%f (changed %f)" % (ll, diff)
            
            if diff < convergence:
                break
            
        
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
        
    return topics

if __name__ == '__main__':
    sys.exit(main())
