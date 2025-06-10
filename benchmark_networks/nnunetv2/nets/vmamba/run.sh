#!/bin/bash

ailb_cluster=("aimagelab-srv-10" "ajeje" "carabbaggio" "ailb-login-01" "ailb-login-02" "aimagelab-srv-00")

current_hostname=$(hostname)


slurm_partition="all_usr_prod"
slurm_account="grana_maxillo"
slurm_time="00:10:00"
slurm_gres="gpu:1"
slurm_constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G"
s_constraint="gpu_2080Ti_11G"

random_string=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
sbatch_file="/tmp/$random_string.sbatch"
echo "#!/bin/bash" > $sbatch_file
echo "#SBATCH --job-name=VMamba" >> $sbatch_file
echo "#SBATCH --output=/work/grana_maxillo/Mamba3DMedModels/umamba/nnunetv2/nets/vmamba/file.out" >> $sbatch_file
echo "#SBATCH --error=/work/grana_maxillo/Mamba3DMedModels/umamba/nnunetv2/nets/vmamba/file.err" >> $sbatch_file
echo "#SBATCH --time=$slurm_time" >> $sbatch_file
echo "#SBATCH --mem=32G" >> $sbatch_file
echo "#SBATCH --cpus-per-task=4" >> $sbatch_file
echo "#SBATCH --gres=$slurm_gres" >> $sbatch_file
echo "#SBATCH --partition=$slurm_partition" >> $sbatch_file
echo "#SBATCH --account=$slurm_account" >> $sbatch_file
echo "#SBATCH --constraint=$s_constraint" >> $sbatch_file

echo "source /work/grana_maxillo/Mamba3DMedModels/venv/bin/activate" >> $sbatch_file
echo "module unload cuda/12.1" >> $sbatch_file
echo "module load cuda/11.8" >> $sbatch_file
echo "python main.py" >> $sbatch_file

sbatch $sbatch_file
echo "Submitted sbatch file $sbatch_file"
