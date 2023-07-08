#!/bin/bash
#SBATCH --time=7-0:0:0 
#SBATCH --account=def-skal 
#SBATCH --ntasks 64
#SBATCH --mem 2009GB
#SBATCH --nodes 1
#SBATCH --mail-user=sareeves@uwaterloo.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --array=[0,1,14,18,20,21,23,73,74,78,79,80,81,94,105,132,133,182,183,185,187,188,189,192,195,196,202,203,206,209,214,218,235,237,243,263,267,272,273,280,281,285,287,294,298,303,307,308,311,313,317,319,320,366,417,420,429,431,437,439,442,443,445,446,463,531,533,535,564,572,574,575,587,590,595,626,627,628,632,635,636,638,644,648,649,650,652,655,658,661,662,663,665,666]

# the above sequence is the same as the one used for rosetta relaxation

module purge

module load hmmer/3.2.1
IFS=','
text=$(cat s669_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
folder=${strarr[0]}; echo $folder
cd $folder

name=$(find ../../fast/sequences/fasta_up -name $folder'_[0-9A-Z].fa')
th=$(cat $name | head -n 2 | tail -n 1 | wc -c) 
th=$(( th / 2 ))
echo $th
echo ${folder}_MSA

jackhmmer --cpu 16 -A ${folder}_MSA -N 8 $name ~/projects/def-skal/sareeves/uniref100.fasta > jackhmmer_log.txt
