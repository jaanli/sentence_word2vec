#!/bin/bash

# preprocess text8 data to test sentence-level embeddings

# get the regular data
wget http://mattmahoney.net/dc/text8.zip -O text8.zip
unzip text8.zip
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm source-archive.zip

# split into sentences
cp text8 text8_split
# add newlines after every 'the' just for debugging
perl -i -pe 's/the/\nthe/g' text8_split
# remove lines that have a single word, because they will have no context
# may need to install gawk
gawk -iinplace 'NF>=2' text8_split
