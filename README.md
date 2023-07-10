## General Setup

We provide the processed predictions for FireProtDB and S461 in `./data/fireprot_mapped_preds.csv` and `./data/s461_mapped_preds.csv`, respectively.
However, to reproduce the predictions you can follow the below sections for preprocessing and inference. We also provide the pre-extracted features
for analysis in the corresponding `./data/{dataset}_mapped_feats.csv` files, but you can reproduce those according to the feature analysis section.

Clone the repository:

`git clone https://github.com/skalyaanamoorthy/thermostability-transfer.git`

`cd thermostability-transfer`

Make a new virual environment:

`virtualenv pslm -p python3.9`

`source pslm/bin/activate`

## Inference Setup

If you have a sufficient NVIDIA GPU (tested on 3090 and A100) you can make predictions with the deep learning models.

Start by installing CUDA if you have not already: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Then install Pytorch according to the instructions: https://pytorch.org/get-started/locally/

Now you can install the requirements:

`pip install -r requirements.txt`

Finally, install evcouplings with no dependencies (it is an old package which will create conflicts):

`pip install evcouplings --no-deps`

You will also need to install the following inference repositories if you wish to use these specific models:

ProteinMPNN:

`git clone https://github.com/dauparas/ProteinMPNN`
	
Note: ProteinMPNN directory will be used as input for ProteinMPNN scripts; it will need to be specify

Tranception:

`https://github.com/OATML-Markslab/Tranception`

Follow the instructions in the repo to get the Tranception_Large (parameters) binary and config. You do not need to the setup the conda environment.

## Preprocessing Setup

In order to perform inference you will first need to preprocess the structures and sequences. Follow the above instructions before proceeding.

You will need the following additional tools for preprocessing:

Modeller (for repairing PDB structures): https://salilab.org/modeller/download_installation.html

(you will need a license, which is free for academic use)

To make modeller visible to the Python scripts, you can append to your `./pslm/bin/activate` file following the pattern shown in `convenience_scripts/append_modeller_paths.sh`:

`sh convenience_scripts/append_modeller_paths.sh`

Ensuring to replace the modeller version and system architecture as required. Then make sure to restart the virtualenv:

`source pslm/bin/activate`

To run inference on either FireProtDB or S669/S461 you will need to preprocess the mutants in each database, obtaining their structures and sequences and modelling missing residues. You can accomplish this with preprocess.py. Assuming you are in the base level of the repo, you can call:

`python3 preprocessing/preprocess.py`

Note that the output dataframe `./data/fireprot_mapped.csv` is already generated, but the other files are not prepared.
It is expected to see the message '507 observations lost from the original dataset' for FireProtDB. Note that you will also have to do this for S669.

## Running Inference

Then, you can run any of the inference scripts in inference scripts. You can use the template calls from cluster_inference_scripts in order to determine the template for calling each method's wrapper script. For instance, to run ProteinMPNN with 0.2 Angstrom backbone noise on FireProtDB:

python inference_scripts/mpnn.py --db_location 'data/fireprot_mapped.csv' --output 'data/fireprot_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'

Note that ProteinMPNN and Tranception require the location where the github repository was installed as arguments.

## Feature Analysis Setup

For analysis based on features, you can compute the features using preprocessing/compute_features.py. Note that the features have been precomputed and appear in `./data/fireprot_mapped_feats.csv`:
You will need the following tools to help recompute features:

AliStat (for getting multiple sequence alignment statistics): https://github.com/thomaskf/AliStat

`git clone https://github.com/thomaskf/AliStat`
`cd AliStat`
`make`

DSSP (for extracting secondary structure and residue accessibility): https://github.com/cmbi/dssp

`sudo apt install dssp`

OR

`git clone https://github.com/cmbi/dssp` and follow instructions.

Finally, you can run the following to compute the features. 

`python3 preprocessing/compute_features.py`

It is expected that there will be some errors in computing features. AliStats might fail for large alignments if you do not have enough RAM. Remember that the features have been pre-computed for your convience as stated above, and any missing features can be handled by merging dataframes.

## Analysis

Then, you can use the analysis_notebooks to reproduce the figures, modifying the path(s) at the start of the file and running each cell.



