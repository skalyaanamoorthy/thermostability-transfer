## Protein Sequence Likelihood Modelling for Thermostability Prediction

This repository is for facilitating access to self-supervised deep learning models, which predict the likelihood of amino acids in their biochemical context, in order to make zero-shot predictions of thermostability measurements.

## General Setup

We provide the processed predictions for FireProtDB and S461 in `./data/fireprot_mapped_preds.csv` and `./data/s461_mapped_preds.csv`, respectively. However, to reproduce the predictions you can follow the below sections for preprocessing and inference. We also provide the pre-extracted features for analysis in the corresponding `./data/{dataset}_mapped_feats.csv` files, but you can reproduce those according to the feature analysis section.

Clone the repository:

`git clone https://github.com/skalyaanamoorthy/thermostability-transfer.git`

`cd thermostability-transfer`

Make a new virual environment (tested with Python=3.8+). On a cluster, you might need to `module load python` first:

`virtualenv pslm`

`source pslm/bin/activate`

## Inference Setup

If you have a sufficient NVIDIA GPU (tested on 3090 and A100) you can make predictions with the deep learning models.

Start by installing CUDA if you have not already: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html. At time of writing you will need to get CUDA 11.X in order to be able to install the torch-* requirements. If you are on a cluster, make sure you have the cuda module loaded e.g. `module load cuda` as well as any compiler necessary e.g. `module load gcc`. If you are using WSL2, you should be able to just use `sh ./convenience_scripts/cuda_setup_wsl.sh`. MIF-ST also requires cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

Then install Pytorch according to the instructions: https://pytorch.org/get-started/locally/ . In most cases, it will suffice to `pip install torch`.

On the ComputeCanada cluster, you will have to comment out arrow dependency and load the module instead with `module load arrow`.

Now you can install the requirements:

`pip install -r requirements.txt`

Finally, install evcouplings with no dependencies (it is an old package which will create conflicts):

`pip install evcouplings --no-deps`

You will also need to install the following inference repositories if you wish to use these specific models:

ProteinMPNN:

`git clone https://github.com/dauparas/ProteinMPNN`
	
Note: ProteinMPNN directory will be used as input for ProteinMPNN scripts; it will need to be specified when calling the Python script (--mpnn_loc).

Tranception:

`git clone https://github.com/OATML-Markslab/Tranception`

Follow the instructions in the repo to get the Tranception_Large (parameters) binary and config. You do not need to the setup the conda environment.
Again, you will need to specify the location of the repository (--tranception_loc) and the model weights (--checkpoint).

KORPM:

Make sure to have Git LFS in order to obtain the potential maps used by KORPM, otherwise you can download the repository as a .zip and extract it.

`git clone https://github.com/chaconlab/korpm`

You will need to compile korpm with the gcc compiler:

`cd korpm/sbg`

`sh ./compile_korpm.sh`

Like the above methods, there is a wrapper script in inference_scripts where you will need to specify the installation directory with the argument --korpm_loc.

## Preprocessing Setup

In order to perform inference you will first need to preprocess the structures and sequences. Follow the above instructions before proceeding.

You will need the following additional tools for preprocessing:

Modeller (for repairing PDB structures): https://salilab.org/modeller/download_installation.html

You will need a license, which is free for academic use; follow the download page instructions to make sure it is specified. You can install with conda, but be sure the change the paths in the following script.
To make modeller visible to the Python scripts, you can append to your `./pslm/bin/activate` file following the pattern shown in `convenience_scripts/append_modeller_paths.sh`:

`sh convenience_scripts/append_modeller_paths.sh`

Ensuring to replace the modeller version and system architecture as required. Then make sure to restart the virtualenv:

`source pslm/bin/activate`

To run inference on either FireProtDB or S669/S461 you will need to preprocess the mutants in each database, obtaining their structures and sequences and modelling missing residues. You can accomplish this with preprocess.py. Assuming you are in the base level of the repo, you can call the following (will use the raw FireProtDB obtained from https://loschmidt.chemi.muni.cz/fireprotdb/ Browse Database tab):

`python preprocessing/preprocess.py --dataset fireprotdb`

Note that the output dataframe `./data/fireprot_mapped.csv` is already generated, but the other files are not prepared.
It is expected to see the message '507 observations lost from the original dataset' for FireProtDB. Note that you will also have to do this for S669. S461 is a subset of S669, so you can call either dataset for the `--dataset` argument, and the same preprocessing will occur; the subset will be generated in the analysis notebook (see Analysis section). For inverse/reversion mutations on S669/461, you will use the predicted mutant structures in the structures_mut folder, which we obtained from the authors (thank you, Drs. Birolo and Fariselli): https://academic.oup.com/bib/article/23/2/bbab555/6502552. They will have to be preprocessed as well . We also obtained the original data file Data_s669_with_predictions.csv from the Supplementary information of this paper, adjusting one record to accurately reflect the structure. Citation: Pancotti, C. et al. Predicting protein stability changes upon single-point mutation: a thorough comparison of the available tools on a new dataset. Briefings in Bioinformatics 23, bbab555 (2022).

`python preprocessing.py --dataset s669`

## Running Inference

Then, you can run any of the inference scripts in inference scripts. You can use the template calls from cluster_inference_scripts in order to determine the template for calling each method's wrapper script (they are designed to be called from the cluster_inference_scripts directory, though). On the other hand, to run ProteinMPNN from the repository root with 0.2 Angstrom backbone noise on FireProtDB:

`python inference_scripts/mpnn.py --db_location 'data/fireprot_mapped.csv' --output 'data/fireprot_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'`

Again, note that you must specify the install location for ProteinMPNN, Tranception, and KORPM because they originate from repositories.

If you are running on a cluster, you will likely find it convenient to modify the `cluster_inference_scripts` and directly submit them; they are designed to be submitted from their own folder as the working directory, rather than the root of the repo like all other files. Note that ESM methods (ESM-1V, MSA-Transformer, ESM-IF) and MIF methods (MIF and MIF-ST) will require substantial storage space and network usage to download the model weights on their first run (especially ESM-1V). To run inference of inverse/reversion mutations for structural methods you will need the predicted mutants as stated above, and you will have to use the _inv versions of each structural method.

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

`python3 preprocessing/compute_features.py --alistat_loc YOUR_ALISTAT_INSTALLATION`

It is expected that there will be some errors in computing features. AliStats might fail for large alignments if you do not have enough RAM. Remember that the features have been pre-computed for your convience as stated above, and any missing features can be handled by merging dataframes.

## Analysis

Then, you can use the analysis_notebooks to reproduce the figures, modifying the path(s) at the start of the file and running each cell.



