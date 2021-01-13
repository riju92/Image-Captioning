#!/bin/bash
#SBATCH --job-name=imgCaption                 # Job name
#SBATCH --mail-type=END,FAIL         	    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rishabh.das@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --mem=32gb                          # Job memory request
#SBATCH --output=run_data_script_log%j.log  # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:2
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --account=cis6930              # Assign group
#SBATCH --qos=cis6930

pwd; hostname; date

module load python
module load cuda
module load tensorflow/2.3.1

echo "Running plot script on a single CPU core"


rm run_script_log.txt


python main.py --cache_inception_features True >> run_script_log.txt
