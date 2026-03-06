#!/bin/bash

wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz
# wget http://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz

wget http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz

wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz
#wget http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz -- This is unusued in this analysis

for f in *.gz;
do
    tar -xzvf $f
    rm -f $f
done