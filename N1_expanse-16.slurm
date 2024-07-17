#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.16.2000.100000.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm

srun --mpi=pmi2 -n 16 ./apf -n 1800 -i 100000 -x 1 -y 16
srun --mpi=pmi2 -n 16 ./apf -n 1800 -i 100000 -x 1 -y 16 -k
srun --mpi=pmi2 -n 32 ./apf -n 1800 -i 100000 -x 2 -y 16 
srun --mpi=pmi2 -n 32 ./apf -n 1800 -i 100000 -x 2 -y 16 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 8 -y 8 
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 8 -y 8 -k
srun --mpi=pmi2 -n 128 ./apf -n 1800 -i 100000 -x 1 -y 128
srun --mpi=pmi2 -n 128 ./apf -n 1800 -i 100000 -x 2 -y 64
srun --mpi=pmi2 -n 128 ./apf -n 1800 -i 100000 -x 4 -y 32
srun --mpi=pmi2 -n 128 ./apf -n 1800 -i 100000 -x 8 -y 16
srun --mpi=pmi2 -n 128 ./apf -n 1800 -i 100000 -x 8 -y 16 -k


srun --mpi=pmi2 -n 128 ./apf -n 8000 -i 8000 -x 8 -y 16
srun --mpi=pmi2 -n 128 ./apf -n 8000 -i 8000 -x 8 -y 16 -k

srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 16 -y 16 
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 16 -y 16 -k

srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 8 -y 24 
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 8 -y 24 -k
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16 -k



srun --mpi=pmi2 -n 384 ./apf -n 8000 -i 8000 -x 8 -y 48
srun --mpi=pmi2 -n 384 ./apf -n 8000 -i 8000 -x 16 -y 24
srun --mpi=pmi2 -n 384 ./apf -n 8000 -i 8000 -x 16 -y 24 -k