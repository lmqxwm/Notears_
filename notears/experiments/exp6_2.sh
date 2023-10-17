#!/bin/bash
 
#---------------------------------------------------------------------------------
# Account information
 
#SBATCH --account=pi-naragam              # basic (default), phd, faculty, pi-<account>
 
#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=10          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=8G           # requested memory
#SBATCH --time=7-00:00:00          # wall clock limit (d-hh:mm:ss)

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=exp6    # user-defined job name

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.10

#---------------------------------------------------------------------------------
# Commands to execute below...

python3 exp6_2.py