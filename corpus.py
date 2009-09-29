#! /usr/bin/env python

from __future__ import with_statement

import os
import os.path
import re
import sys
import collections
#import numpy


SPLITTER = re.compile("[^a-zA-Z0-9_\']+")

class Vocabulary(list):
    """docstring for ClassName"""
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.types = dict()

    def add(self, token):
        """add a token to the vocabulary"""
        if token not in self.types:
            self.types[token] = len(self)
            self.append(token)

    def index(self, token):
        """look up index for type"""
        return self.types[token]
        
    def __str__(self):
        return " ".join(self)
        
        
class Document(object):
    """a document"""
    
    def __init__(self, vocab, label):
        self.vocabulary = vocab
        self.label = label
        self.counts = collections.defaultdict(int)
    
    def readfile(self, filename):
        """read document from file"""
        self.docname = os.path.basename(filename)
        with open(filename) as f:
            for line in f.xreadlines():
                self.__parseline(line.strip())
        
    def __parseline(self, line):
        try:
            for token in SPLITTER.split(line):
                token = token.strip().lower()
                if token:
                    self.counts[token] += 1
                    self.vocabulary.add(token)
        except: 
            print "error: ", line

    def name(self):
        """docstring for name"""
        return "%s/%s" % (self.label, self.docname)
        
    def __str__(self):
        """dump out the vocabulary and frequency counts"""
        return "%s/%s => %s" % (
            self.label, 
            self.docname,
            " ".join(["%s:%d" % (word, count) for (word, count) in self.counts.iteritems()]))
        # print self.counts.sorted()
    
    def __getitem__(self, key):
        """Delegate to self.counts"""
        return self.counts[key]
    
    

class Corpus(list):
    """docstring for Corpus"""
    def __init__(self):
        super(Corpus, self).__init__()
        self.vocabulary = Vocabulary()
                         
    def readfiles(self, directory):
        """docstring for readfiles"""
        label = os.path.basename(directory)
        for doc_filename in os.listdir(directory):
            if doc_filename[0] != '.':
                doc_path = "%s/%s" % (directory, doc_filename)
                if (os.path.isdir(doc_path)):
                    self.readfiles(doc_path)
                else:
                    doc   = Document(self.vocabulary, label)
                    doc.readfile(doc_path)
                    # print doc
                    self.append(doc)
   
         

if __name__ == '__main__':
    
    def main(argv=None):
        if argv is None:
            argv = sys.argv

        doc_directory = ""

        if len(argv) > 1:
            doc_directory = argv[1]

        corpus = Corpus()
        corpus.readfiles(doc_directory)
        print corpus.vocabulary

        print [corpus.vocabulary.index(t) for t in corpus.vocabulary]
        
    sys.exit(main())
