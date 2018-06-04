#!/bin/bash
# Sampling of none-frequent graphs and mining of discriminative subgraphs:
k=1
while [ $k -le 10 ]
do
  python 'sample_gsp_infreq.py' $k
  l=1
  while [ $l -le 200 ]
    do
      ./gspan 100 'train_data//freq_training_'$k'.gsp' 'train_data//infreq_training_'$k'_'$l'.gsp' ig > 'subgraphs//top_sg_'$k'_'$l'.gsp'
      ((l++))
  done
  ((k++))
done

