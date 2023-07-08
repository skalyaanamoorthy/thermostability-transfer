Make a new virual environment:

`virtualenv pslm`

Install Pytorch according to the instructions (you might also need to install cuda): https://pytorch.org/get-started/locally/

torch-* must be installed after torch:

`pip install torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv`

Now you can install the requirements with no additional dependencies:

`pip install -r requirements.txt`

You will also need to install the following inference repositories if you wish to use them:

ProteinMPNN:

`git clone https://github.com/dauparas/ProteinMPNN`
	
 note: ProteinMPNN directory will be used as input for ProteinMPNN scripts

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
and follow instructions
