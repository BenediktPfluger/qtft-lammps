#!/bin/bash -l
# Slurm job of a qtft-lammps run on the LRZ Linux cluster, one array task per replica (rep folders
# in replicas.txt). Filled by qtft/build.py from job_template.slurm; submit from the run folder.
#SBATCH --job-name={{job_name}}
#SBATCH --clusters={{cluster}}
#SBATCH --partition={{partition}}
#SBATCH --qos={{qos}}
#SBATCH --ntasks={{mpi_procs}}
#SBATCH --cpus-per-task=1
#SBATCH --mem={{slurm_mem}}
#SBATCH --time={{slurm_time}}
#SBATCH --array=1-{{n_tasks}}{{array_limit}}
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-type={{mail_type}}
#SBATCH --mail-user={{mail_user}}
#SBATCH --export=NONE

# login shell (-l): LRZ's base setup provides module and the Intel MPI settings
module load {{modules}}
export SLURM_EXPORT_ENV=ALL     # srun passes this environment on; --export=NONE would pass an empty one
export OMP_NUM_THREADS=1

rep=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SLURM_SUBMIT_DIR/replicas.txt")
cd "$SLURM_SUBMIT_DIR/$rep" || exit 1
echo "$SLURM_JOB_ID" > pid      # replica has started (as the pid file of a local run)
echo running > status
# --mpi=pmi2: the serial cluster starts no PMI by default, Intel MPI needs PMI2
srun --mpi=pmi2 {{lmp}} -in in.lammps -log log.lammps -screen none > lammps.out 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q QTFT_DONE log.lammps; then
    echo done > status
else
    echo failed > status
    exit 1
fi
