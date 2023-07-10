Clone the repository:

`git clone https://github.com/skalyaanamoorthy/thermostability-transfer.git`

`cd thermostability-transfer`

Make a new virual environment:

`virtualenv pslm -p python3.9`

`source pslm/bin/activate`

Install Pytorch according to the instructions (you might also need to install cuda): https://pytorch.org/get-started/locally/

torch-* must be installed after torch:

`pip install torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv`

Now you can install the requirements with no additional dependencies:

`pip install -r requirements.txt`

You will also need to install the following inference repositories if you wish to use them:

ProteinMPNN:

`git clone https://github.com/dauparas/ProteinMPNN`
	
 Note: ProteinMPNN directory will be used as input for ProteinMPNN scripts

Tranception:

`https://github.com/OATML-Markslab/Tranception`

Follow the instructions in the repo to get the Tranception_Large (parameters) binary and config

You will need the following for preprocessing:

Modeller: https://salilab.org/modeller/download_installation.html

-make sure it is accessible in the virtual environment
- e.g. follow installation instructions, can add pythonpath and ld_library path to activate script
-you will need a license to use it (free for academic use)

You will need the following for computing features:

AliStat:

`git clone https://github.com/thomaskf/AliStat`
`cd AliStat`
`make`
This sequence should work, but see the instructions on the repo if not

DSSP

`sudo apt install dssp`
OR
`git clone https://github.com/cmbi/dssp` 
and follow instructions.

To run inference on either FireProtDB or S669/S461 you will need to preprocess the mutants in each database, obtaining their structures and sequences and modelling missing residues. You can accomplish this with preprocess.py. Assuming you are in the base level of the repo, you can call:

`python3 preprocessing/preprocess.py --db_loc ./data/fireprotdb_results.csv -o .`

Note that you will also have to do this for S669.

Then, you can run any of the inference scripts in inference scripts. You can use the template calls from cluster_inference_scripts in order to determine the template for calling each method's wrapper script. For instance, to run ProteinMPNN with 0.2 Angstrom backbone noise on FireProtDB:

python inference_scripts/mpnn.py --db_location 'data/fireprot_mapped.csv' --output 'data/fireprot_mapped_preds.csv' --mpnn_loc ~/software/ProteinMPNN --noise '20'

Note that ProteinMPNN and Tranception require the location where the github repository was installed as arguments



