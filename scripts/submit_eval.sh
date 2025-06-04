for filename in sbatch_eval/*.sh; do
    # echo $filename
    sbatch $filename
done