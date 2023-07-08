#!/bin/bash
#SBATCH --time=2:00:0
#SBATCH --account=rrg-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem 64GB
#SBATCH --output=%A-%x

module purge
module load python
module load scipy-stack
module load StdEnv/2020  gcc/9.3.0  cuda/11.4

source ~/scratch/dl3/bin/activate

if ! test -e './s669_mapped_preds.csv'; then cp './s669_mapped.csv' './s669_mapped_preds.csv'; fi
python mpnn.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --dataset 's669' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.01'
python mpnn.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --dataset 's669' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.02'
python mpnn.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --dataset 's669' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.03'
python mpnn.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --dataset 's669' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.04'
python mpnn.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --dataset 's669' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.05'
if ! test -e './fireprot_mapped_preds.csv'; then cp './fireprot_mapped.csv' './fireprot_mapped_preds.csv'; fi
python mpnn.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --dataset 'fireprot' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.01'
python mpnn.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --dataset 'fireprot' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.02'
python mpnn.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --dataset 'fireprot' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.03'
python mpnn.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --dataset 'fireprot' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.04'
python mpnn.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --dataset 'fireprot' --mpnn_loc ~/scratch/ProteinMPNN --noise '20' --bbnoise '0.05'
