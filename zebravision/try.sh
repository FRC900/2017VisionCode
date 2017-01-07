#!/bin/bash

rm *

FILE=$(ls -t /home/marshall/Downloads/2016* | head -1)

cp $FILE .

tar -zxvf $FILE

mv snap* network.caffemodel

cd ../

./zv
