#!/bin/bash
#SBATCH --time=7-0:0:0
#SBATCH --account=def-skal
#SBATCH --mem=2009GB
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --array=[0,1,14,18,20,21,23,73,74,78,79,80,81,94,105,132,133,182,183,185,187,188,189,192,195,196,202,203,206,209,214,218,235,237,243,263,267,272,273,280,281,285,287,294,298,303,307,308,311,313,317,319,320,366,417,420,429,431,437,439,442,443,445,446,463,531,533,535,564,572,574,575,587,590,595,626,627,628,632,635,636,638,644,648,649,650,652,655,658,661,662,663,665,666]

module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack
#source ~/projects/rrg-skal/sareeves/dl/bin/activate

### The below section looks up information on the protein like the chain and mutation of interest
IFS=','
text=$(cat s669_unique_muts_offsets.csv | head -n $((${SLURM_ARRAY_TASK_ID}+2)) | tail -n 1)
read -a strarr <<< "$text"
folder=${strarr[0]}; echo $folder
cd $folder
name=$folder'_MSA'

#if test -f ${name}_cov75_id90.a3m; then
#	echo "skipping because filtered already exists"
#	cd ..
#	continue
#fi

perl ~/projects/def-skal/sareeves/scripts/reformat.pl sto a3m $name ${name}.a3m -r > log_msat.txt 
hhfilter -i ${name}.a3m -o ${name}_cov75_id90.a3m -cov 75 -id 90 >> log_msat.txt
