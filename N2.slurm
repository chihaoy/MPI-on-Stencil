#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.16.2000.100000.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=64G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:01:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xix025@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu/0.15.4
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm

srun --mpi=pmi2 -n 128 ./apf -n 8000 -i 8000 -x 1 -y 128
# srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 2 -y 96 -k
# srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48
# srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48 -k
# srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16
# srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16 -k

