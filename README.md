# PrePeP - A Tool for the Identification and Characterization of Pan Assay Interference Compounds
The code for the paper submitted for the KDD'18 conference (Applied Data Science Track): http://www.kdd.org/kdd2018/

## Data preparation

- **split_num.py** - split *.sdf* -> *frequent.sdf*, *infrequent.sdf* and extraction of numerical features
- **sdf2gsp_nolabels_mod.pl** - perl script for converting *.sdf* -> *.gsp*
- **remove_hydrogens.py** - for removing hydrogen atoms from the data

## Sampling

- **sample_gsp_freq.py** - sampling FHs and corresponding numerical features into training and test sets
- **construct.sh** - sampling iFHs with corresponding numerical features into training and test sets (uses 'sample_gsp_infreq.py') and discriminating subgraphs mining (uses 'gspan')

## Encoding and learning

- **encoding.py** - creating feature vectors for one experiment (1:1, 1:10 or 1:100 ratio of FH:iFH for test set) by encoding train and test data with discriminatory subgraphs - most time consuming part (may take a week for a run in the most extream setting)
- **learning.py** - model learning and validation

## Evaluation

- **construct_all.sh** - sampling iFHs for training (uses *sample_gsp_infreq_all.py*) and discriminating subgraphs mining (uses *gspan*)
- **classifier.py** - encoding data for a classifier from all FHs and sampled on the previous step iFHs
- **evaluation_encoding.py** - encoding evaluation datasets (RandomPAINS/RandomNoPAINS/DCM) by subgraphs obtained on the previous step (put corresponding *S9_hless.gsp*, *S10_hless.gsp*, *S11_hless.gsp* into *evaluation* subfolder)
- **evaluation.py** - learning a classifier from the data obtained on step 2, performing evaluation, building confusion matrices and histograms (uses activity values in *activity_S9.txt* and *activity_S10.txt* in *evaluation* subfolder)

**System requirements**: Python 2.7, networkx library.

The data is available here: https://zimmermanna.users.greyc.fr/supplementary-material.html

