#! /usr/bin/env python

from __future__ import with_statement

from corpus import *
# import os.path
# import re
# import sys
# import collections
import numpy
import collections


detailed_trace = False


class DocumentTopicAssignment(collections.defaultdict):
    """docstring for DocumentTopicAssignment"""
    
    def __init__(self, doc, numTopics):

        def random_vector():
            """initialize a random vector"""
            vec = numpy.random.random((numTopics))
            vec /- vec.sum()
            return vec

        super(DocumentTopicAssignment, self).__init__(random_vector)
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
        
        self.gamma = [DocumentTopicAssignment(doc, numTopics) for doc in corpus]    # posterior
            # gamma is (j,w,k)
            
        self.phi   = numpy.zeros((self.numTypes, self.numTopics))
        self.theta = numpy.zeros((self.numDocs,  self.numTopics))

        #self.gamma = numpy.ones((self.numDocs, self.numTypes, self.numTopics))
        # self.gamma = numpy.random.random((self.numDocs, self.numTypes, self.numTopics))
        # self.phi   = numpy.zeros((self.numTypes, self.numTopics))
        # self.theta = numpy.zeros((self.numTopics, self.numDocs))

        # for d in self.gamma:
        #     for w in d:
        #         w /= w.sum()

        # def countArray(doc):
        #     return numpy.array([doc.counts[w] for w in corpus.vocabulary])
        # 
        # self.observations = numpy.array([countArray(doc) for doc in corpus])

    def expectation(self):
        """calculate expected phi and theta values, returning N_wk, N_kj, and N_k.
           Actually, we don't really need to calculate phi and theta, since we only 
           need the Ns for the M step...."""
        
        numTopics = self.numTopics
           
        def zero_vector():
            """initialize a zero vector"""
            return numpy.zeros((numTopics))

        n_wk   = collections.defaultdict(zero_vector)  #numpy.zeros((self.numTypes, self.numTopics))
        n_kj   = collections.defaultdict(zero_vector) 

        n_k    = numpy.zeros((self.numTopics))
        n_j    = numpy.zeros((self.numTopics))
        
        for j in xrange(self.numDocs):
            for word in self.corpus.vocabulary:
                p_topic = self.gamma[j][word] * self.corpus[j][word]
                
                if detailed_trace:
                    print "p_topic %3d %20s %s" % (j, word, p_topic)
                    
                n_wk[word] += p_topic
                n_kj[j]    += p_topic
                n_k        += p_topic
        
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            p = n_wk[word]/n_k
            p /= p.sum()
            self.phi[w] = p

        for j in xrange(self.numDocs):
            n_j = n_kj[j].sum()
            self.theta[j] = n_kj[j]/n_j

            
        # n_wj      = self.observations.transpose()
        # gamma_wjk = self.gamma.transpose((1,0,2))
        
        # n_wk = numpy.array([numpy.dot(n_wj[w], gamma_wjk[w]) for w in xrange(self.numTypes)])
        # n_k  = n_wk.sum(axis=0)
        
        # self.phi = n_wk/n_k
        # for w in self.phi:
        #     w /= w.sum()

        return (n_wk, n_kj, n_k)

    # def expectation_theta(self):
    #     """calculate expected theta values, returning N_kj"""
    #     # n_jw      = self.observations
    #     # gamma_jwk = self.gamma
    #     
    #     n_kj   = collections.defaultdict(float)  #numpy.zeros((self.numTypes, self.numTopics))
    #     
    #     for w in self.corpus.vocabulary:
    #         for j in xrange(self.numDocs):
    #             n_kj[j] += gamma[j][w]
    #     
    #     n_kj = numpy.array([numpy.dot(self.observations[j], self.gamma[j]) for j in xrange(self.numDocs)])
    #     n_j  = n_kj.sum(axis=0)
    #     
    #     self.theta = n_kj/n_j
    #     for j in self.theta:
    #         j /= j.sum()
    #     
    #     return n_kj

    def maximize_gamma(self, n_wk, n_kj, n_k):
        """update gamma using new param values"""
        for j in xrange(self.numDocs):
            for word in self.corpus.vocabulary:
                w = self.corpus.vocabulary.index(word)
                g = self.phi[w] * self.theta[j]
                g /= g.sum()
                self.gamma[j][word] = g
                # self.gamma[j][word] = n_wk[word] * n_kj[j] / n_k
                # self.gamma[j][word] /= self.gamma[j][word].sum()


    def em(self):
        """docstring for em"""
        
        (n_wk, n_kj, n_k) = self.expectation()
        
        if detailed_trace:
            print "n_wk:"
            for word in self.corpus.vocabulary:
                print "\t%20s %s" % (word, n_wk[word])
            
            print "n_kj:"
            for (j, p_topic) in n_kj.iteritems():
                print "\t%20s %s" % (j, p_topic)
            
            print "n_k:\t", n_k
        
        self.maximize_gamma(n_wk, n_kj, n_k)

        
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

    def iterate(self, numIterations=1, startIteration=1):
        """docstring for iterate"""

        for i in xrange(numIterations):
            print "iteration %d: " % (i+startIteration)
            self.em()
            self.printPhi()
            self.printTheta()
            if detailed_trace:
                self.printGamma()
        
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
    
    # topics.printGamma()
    
    # for i in xrange(20):
    #     print "iteration %d: " % i
    #     topics.iterate()
    
    return topics

if __name__ == '__main__':
    sys.exit(main())
