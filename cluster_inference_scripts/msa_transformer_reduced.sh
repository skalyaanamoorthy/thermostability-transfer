#!/bin/bash
#SBATCH --time=10:0:0
#SBATCH --account=def-skal
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --output=%A-%x

module load hh-suite/3.3.0
module load python/3.8
module load scipy-stack
source ~/projects/def-skal/sareeves/dl2/bin/activate

#1AQH_A 1BYW_A 1FEP_A 1HYN_P
#ro_folders=$(echo predictions/[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]_[0-9A-Z]/ | tac -s ' ')
for folder in 1QGD_A 1QQV_A 1WIT_A
#1KCQ_A 1LVM_A 1IR3_A 1LVM_A 1UZC_A 1X0J_A 2DVV_A 2LTB_A 3S92_A 
do
	#if [[ "$folder" > "predictions/1TDI_A" ]]; then
	folder=predictions/$folder
	cd $folder
        pdb=$(echo $folder | awk -F '/' '{print $2}' | awk -F '_' '{print $1}')
        chain=$(echo $folder | awk -F '/' '{print $2}' | awk -F '_' '{print $2}')
        echo $folder $pdb $chain
	
	seqfile='../../windows/'$pdb'_'$chain'.fa'
        seq=$(sed -n '2{p;q;}' $seqfile)

        st=$(awk -F ',' '{print $1}' '../../windows/'$pdb'_'$chain)
        ed=$(awk -F ',' '{print $2}' '../../windows/'$pdb'_'$chain)
        echo $st $ed

        to_run=()
	dms='../../DMS_MSA/'$pdb'_'$chain'_fireprot.csv'
        if [[ -e $dms ]]; then to_run+=('fireprot'); fi
        dms='../../DMS_MSA/'$pdb'_'$chain'_s669.csv'
        if [[ -e $dms ]]; then to_run+=('s669'); fi
        echo ${to_run[@]}

        for dataset in ${to_run[@]}
	do
		dms='../../DMS_MSA/'$pdb'_'$chain'_'$dataset'.csv'
		echo 'Attempting to run dataset found in' $dms
		if [[ $dataset == 's669' ]]; then
			if [[ $( ls *expanded*.a3m | wc -l ) -gt 0 ]]
			then name=$( ls *expanded*.a3m | head -n 1 | awk -F '.' '{print $1}' )
			else name=$( ls *.a3m | head -n 1 | awk -F '.' '{print $1}' )
			fi
		else name=$( ls *.a3m | head -n 1 | awk -F '.' '{print $1}' )
		fi
		echo $name

		# get a reduced alignment which is limited to the window columns
		cut -c $(($st+1))-$((ed+1)) $name.a3m > $name.tmp
		if [[ $st -gt 0 ]]; then
			echo 'adding in line indicators'
			awk -v ln=1 '(NR+1)%2 {print ">" ln++ "\n" $0 }' $name.tmp > $name'_reduced.a3m'
		else
			cat $name.tmp > $name'_reduced.a3m'
        	fi
		rm $name.tmp

		python ../../subsample_fix.py --infile $name'_reduced.a3m' > sampling.txt
	
		runtime=$((end-start))
		echo $runtime >> log_msat.txt

		start=`date +%s`
		for i in 0 1 2 3 4
		do
			python ~/projects/def-skal/sareeves/esm/examples/variant-prediction/predict.py --model-location esm_msa1b_t12_100M_UR50S --sequence $seq --dms-input $dms --mutation-col mutation --dms-output ${name}'_msat_'$dataset'_'${i}'_scores.csv' --offset-idx $((st+1)) --scoring-strategy masked-marginals --msa-path ${name}_reduced_subsampled_${i}.a3m --msa-samples 192 >> log_msat.txt || break
		done 
		end=`date +%s`
		runtime=$((end-start))
		echo $runtime > runtime_msat_$dataset.txt
	done
	cd ../..
done
