#!/bin/bash

# wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz -- This is unusued in this project
# wget http://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz

wget http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz

wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz
#wget http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz -- This is unusued in this project

for f in *.gz;
do
    tar -xzvf $f
    rm -f $f
done