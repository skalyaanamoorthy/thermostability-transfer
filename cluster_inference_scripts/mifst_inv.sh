#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --account=def-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem 64GB
#SBATCH --output=%A-%x

module purge
module load python
module load scipy-stack
module load StdEnv/2020  gcc/9.3.0  cuda/11.4

#source ~/projects/def-skal/sareeves/dl2/bin/activate
source ~/scratch/dl3/bin/activate

if ! test -e './s669_mapped_preds.csv'; then cp './s669_mapped.csv' './s669_mapped_preds.csv'; fi
python mif.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --model 'mifst' --inverse > './logs/log_mifst_s669_inv.txt'
#if ! test -e './fireprot_mapped_preds.csv'; then cp './fireprot_mapped.csv' './fireprot_mapped_preds.csv'; fi
#python mif.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --model 'mif' --inverse > './logs/log_mif_fireprot_inv.txt'
