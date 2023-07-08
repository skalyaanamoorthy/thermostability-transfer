#!/bin/bash
#SBATCH --time=48:0:0
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

source ~/scratch/dl3/bin/activate

if ! test -e './s669_mapped_preds.csv'; then cp './s669_mapped.csv' './s669_mapped_preds.csv'; fi
python msa_transformer.py --db_location './s669_mapped.csv' --alignments '../original_subsampled_alignments' --output './s669_mapped_preds.csv' > './logs/log_msat_s669.txt'
if ! test -e './fireprot_mapped_preds.csv'; then cp './fireprot_mapped.csv' './fireprot_mapped_preds.csv'; fi
python msa_transformer.py --db_location './fireprot_mapped.csv' --alignments '../original_subsampled_alignments' --output './fireprot_mapped_preds.csv' > './logs/log_msat_fireprot.txt'
