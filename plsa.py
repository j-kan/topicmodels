#! /usr/bin/env python

import numpy
import collections
from math import log, fabs
from corpus import *


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
        
        self.n_wk   = numpy.random.random((self.numTypes, self.numTopics))
        self.n_kj   = numpy.random.random((self.numDocs, self.numTopics))
        
            # initialize by assigning types to topics randomly
            #  so this is equivalent to multiplying these matrices by the token count for the appropriate dimension
            #  easier to explicitly manage type/topic assignments in 2 parallel arrays for w and z

        self.n_k    = self.n_wk.sum(axis=0)
        self.n_j    = self.n_kj.sum(axis=1)


    def calc_updated_gamma_w_j(self, w, j):
        """calculate updated gamma values for a given word and doc index.
            We can drop n_j because it doesn't depend on k, and we normalize over k"""
        # g = self.phi[w] * self.theta[j]
        g = self.n_wk[w] * self.n_kj[j] / self.n_k
        g /= g.sum()
        return g

    def expectation(self):
        """update gamma using new param values"""
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            for j in xrange(self.numDocs):
                self.gamma[j][word] = self.calc_updated_gamma_w_j(w,j)

                
    def calc_updated_phi_w(self, w):
        """calc the updated phi value for a given word index"""
        p = self.n_wk[w]/self.n_k
        p /= p.sum()
        return p

    def calc_updated_theta_j(self, j):
        """calc the updated theta value for a given doc index"""
        return self.n_kj[j]/self.n_j[j]
        
        
    def maximization(self, trace=False):
        """calculate MLE for phi and theta values"""
        
        self.n_wk.fill(0.0)
        self.n_kj.fill(0.0)

        self.n_k.fill(0.0)
        self.n_j.fill(0.0)
        
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            for j in xrange(self.numDocs):
                if word in self.corpus[j]:
                    p_topic = self.gamma[j][word] * self.corpus[j][word]
                    self.n_wk[w] += p_topic
                    self.n_kj[j] += p_topic
                
                    if trace:
                        print " %3d %20s %s" % (j, word, p_topic)

        self.n_k    = self.n_wk.sum(axis=0)
        self.n_j    = self.n_kj.sum(axis=1)
                
        if trace:
            self.printNs()
        
        for w in xrange(self.numTypes):
            self.phi[w] = self.calc_updated_phi_w(w)

        for j in xrange(self.numDocs):
            self.theta[j] = self.calc_updated_theta_j(j)



    def em(self):
        """docstring for em"""

        self.expectation()
        self.maximization()


    def logLikelihood(self, trace=False):
        """docstring for logLikelihood"""
        l = numpy.dot(self.phi, self.theta.transpose())
        ll = 0.0
        for j in xrange(self.numDocs):
            for w in xrange(self.numTypes):
                word = self.corpus.vocabulary[w]
                n = self.corpus[j][word]
                if n > 0:
                    p = l[(w,j)]
                    ll += n*log(p)
                    if trace:
                        print "\t%d %20s %6.4f * %2d" % (j, word, p, n)
        return ll
        
        

        
    def printNs(self):
        """print N_xx matrices"""
        print "n_wk:"
        for w in xrange(self.numTypes):
            word = self.corpus.vocabulary[w]
            print "\t%20s %s" % (word, self.n_wk[w])
        
        print "n_kj:"
        for j in xrange(self.numDocs):
            print "\t%20s %s" % (j, self.n_kj[j])
        
        print "n_k:\t\t", self.n_k
        print "n_j:\t\t", self.n_j
        
        
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
        self.printGamma()


    def iterate(self, numIterations=1, convergence=0.01, startIteration=1):
        """docstring for iterate"""
        
        prev_ll = 0
        diff = 0
        
        for i in xrange(numIterations):
            print "iteration %3d: " % (i+startIteration),
            self.em()
            ll = self.logLikelihood()
            diff = fabs((ll - prev_ll)/ll)
            prev_ll = ll
            print "%6.4f  (%8.4f %%)" % (ll, diff*100)
            
            if diff < convergence:
                break
        
        return diff
            
    def topic_words(self, numWords=10):
        """find topic words"""

        n_kw = self.n_wk.transpose()
        n_w  = self.n_wk.sum(axis=1)
        
        def top_words(i, vocabulary):
            p_w = n_kw[i]/n_w
            indices = p_w.argsort()[::-1][0:numWords]  # [::-1] == reverse
            return ' '.join([vocabulary[j] for j in indices])
        
        return [ top_words(i, self.corpus.vocabulary) for i in xrange(self.numTopics) ]
        
    # def __str__(self):
    #     """show topic assignments"""
    #     return self.gamma.__str__() 
    

def topic_words(topics, numWords=10):
    """find topic words"""
    
    def top_words(i, vocabulary):
        indices = topics.phi[:,i].argsort()[::-1]  # [::-1] == reverse
        return ' '.join([vocabulary[j] for j in indices[0:numWords]])
    
    return [ top_words(i, topics.corpus.vocabulary) for i in xrange(topics.numTopics) ]

def topic_words_by_count(topics, numWords=10):
    """find topic words"""
    
    def top_words(i, vocabulary):
        indices = topics.n_wk[:,i].argsort()[::-1]  # [::-1] == reverse
        return ' '.join([vocabulary[j] for j in indices[0:numWords]])
    
    return [ top_words(i, topics.corpus.vocabulary) for i in xrange(topics.numTopics) ]
    
    
def entropy(distribution):
    return -sum([p * log(p) for p in distribution if p > 0])

def kl_divergence(d1, d2):
    """docstring for kl_divergence"""
    return sum([p * log(p/q) for (p,q) in zip(d1, d2) if q > 0])
    

def topic_words_entropy(topics, numWords=10):
    """docstring for topic_entropy"""
    # p_wk = topics.n_wk/topics.n_k
    p_k = topics.n_k/topics.n_k.sum()
    kl = numpy.array([kl_divergence(p_kw, p_k) for p_kw in topics.phi])
    
    def top_words(i, vocabulary):
        p = topics.n_wk[:,i]*kl
        indices = p.argsort()[::-1]  # [::-1] == reverse
        return ' '.join([vocabulary[j] for j in indices[0:numWords]])
    
    return [ top_words(i, topics.corpus.vocabulary) for i in xrange(topics.numTopics) ]
            
         
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
    
    if topics.iterate(100, 0.00001) < 0.00001:
        topics.printPhi()
        topics.printTheta()
        print topics.topic_words(10)
        
    return topics
    

if __name__ == '__main__':
    sys.exit(main())
