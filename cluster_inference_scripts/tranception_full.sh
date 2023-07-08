#!/bin/bash
#SBATCH --time=6:0:0
#SBATCH --account=def-skal
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=%A-%x
#[103,160,227,84,217,136,164,66,15,106,49,3,130,1,99,47,113,190,104]

module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack
module load arrow/10.0.1
source ~/projects/def-skal/sareeves/dl2/bin/activate

#python tranception_.py \
#--checkpoint ~/projects/def-skal/sareeves/Tranception_Large \
#--batch_size_inference 1 \
#--full_msa \
#--DMS_data_folder '.' \
#--inference_time_retrieval \
#--num_workers 8 \
#--db_location 's669_mapped.csv' \
#--output 's669_mapped_preds.csv'
#>> logs/log_tranception_s669.txt
#--MSA_weight_file_name './weights/'${name}.npy \
python tranception_.py \
--checkpoint ~/projects/def-skal/sareeves/Tranception_Large \
--batch_size_inference 1 \
--full_msa \
--DMS_data_folder '.' \
--inference_time_retrieval \
--num_workers 8 \
--db_location 'fireprot_mapped.csv' \
--output 'fireprot_mapped_preds.csv'
>> logs/log_tranception_fireprot.txt
