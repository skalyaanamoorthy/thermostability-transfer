#!/bin/bash
#SBATCH --time=0:10:0
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

#source ~/projects/def-skal/sareeves/dl2/bin/activate
source ~/scratch/dl3/bin/activate

if ! test -e './s669_mapped_preds.csv'; then cp './s669_mapped.csv' './s669_mapped_preds.csv'; fi
python mif.py --db_location './s669_mapped.csv' --output './s669_mapped_preds.csv' --model 'mifst' > './logs/log_mifst_s669.txt'
if ! test -e './fireprot_mapped_preds.csv'; then cp './fireprot_mapped.csv' './fireprot_mapped_preds.csv'; fi
python mif.py --db_location './fireprot_mapped.csv' --output './fireprot_mapped_preds.csv' --model 'mifst' > './logs/log_mifst_fireprot.txt'
