#!/bin/bash
# Sampling of none-frequent graphs and mining of discriminative subgraphs:
python 'sample_gsp_infreq_all.py'
l=1
while [ $l -le 200 ]
  do
    ./gspan 100 'freq_hless.gsp' 'classifier//infreq_train_'$l'.gsp' ig > 'classifier//top_sg_'$l'.gsp'
    ((l++))
done

