#! /usr/bin/env python

from __future__ import with_statement

from corpus import *
# import os.path
# import re
# import sys
# import collections
import numpy
import collections
from math import log


detailed_trace = False


class DocumentTopicAssignment(collections.defaultdict):
    """docstring for DocumentTopicAssignment"""
    
    def __init__(self, numTopics):

        def random_vector():
            """initialize a random vector"""
            vec = numpy.random.random((numTopics))
            vec /- vec.sum()
            return vec

        def zero_vector():
            """initialize a zero vector"""
            return numpy.zeros((numTopics))

        super(DocumentTopicAssignment, self).__init__(zero_vector)
        # self.doc = doc
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
        
        self.gamma = [DocumentTopicAssignment(self.numTopics) for i in xrange(self.numDocs)]    # posterior
            # gamma is (j,w,k)
            
        self.phi   = numpy.random.random((self.numTypes, self.numTopics))
        self.theta = numpy.random.random((self.numDocs,  self.numTopics))
        
        for p in self.phi:
            p /= p.sum()
        
        for p in self.theta:
            p /= p.sum()


    def expectation(self):
        """update gamma using new param values"""
        for j in xrange(self.numDocs):
            for word in self.corpus.vocabulary:
                w = self.corpus.vocabulary.index(word)
                g = self.phi[w] * self.theta[j]
                g /= g.sum()
                self.gamma[j][word] = g
                # self.gamma[j][word] = n_wk[word] * n_kj[j] / n_k
                # self.gamma[j][word] /= self.gamma[j][word].sum()
                
    def maximization(self):
        """calculate MLE for phi and theta values"""
        
        n_wk   = numpy.zeros((self.numTypes, self.numTopics))
        n_kj   = numpy.zeros((self.numDocs, self.numTopics))

        n_k    = numpy.zeros((self.numTopics))
        n_j    = numpy.zeros((self.numTopics))
        
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            for j in xrange(self.numDocs):
                p_topic = self.gamma[j][word] * self.corpus[j][word]
                
                if detailed_trace:
                    print "p_topic %3d %20s %s" % (j, word, p_topic)
                    
                n_wk[w] += p_topic
                n_kj[j] += p_topic
                n_k     += p_topic

        if detailed_trace:
            print "n_wk:"
            for w in xrange(self.numTypes):
                word = self.corpus.vocabulary[w]
                print "\t%20s %s" % (word, n_wk[w])
            
            print "n_kj:"
            for (j, p_topic) in n_kj.iteritems():
                print "\t%20s %s" % (j, p_topic)
            
            print "n_k:\t", n_k
        
        for w in xrange(self.numTypes):
            p = n_wk[w]/n_k
            p /= p.sum()
            self.phi[w] = p

        for j in xrange(self.numDocs):
            n_j = n_kj[j].sum()
            self.theta[j] = n_kj[j]/n_j


    def logLikelihood(self):
        """docstring for logLikelihood"""
        l = 0.0
        for doc in self.gamma:
            for w in doc.itervalues():
                l += log(w.sum())
        return l
        
        
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


    def iterate(self, numIterations=1, startIteration=1):
        """docstring for iterate"""

        for i in xrange(numIterations):
            print "iteration %d: " % (i+startIteration)
            self.em()
            print "\t", self.logLikelihood()
            
        
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
