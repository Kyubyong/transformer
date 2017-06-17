# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    de_train = 'corpora/train.tags.de-en.de'
    en_train = 'corpora/train.tags.de-en.en'
    de_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    en_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0005
    logdir = 'logdir'
    
    # model
    maxlen = 10 # Maximum sentence length. alias = T
    hidden_units = 512 # alias = C
    num_blocks = 6
    num_epochs = 200
    num_heads = 8
    
    sanity_check=True
    min_cnt = 100
    
    
