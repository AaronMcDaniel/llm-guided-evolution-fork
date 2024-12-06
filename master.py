import os

PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=LLMTest_Island_{}
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00
#SBATCH -oReport-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12
module load anaconda3

conda activate llmIntegration #ur local environment
conda info

export HF_HOME=/storage/ice1/0/1/gmiao8/.cache/huggingface

python run_improved.py first_test


# Run Python script
python islandIntegration.py --checkpoints {}
"""

def submit_run(tempfile, text):
    with open(tempFile, 'w') as file:
        file.write(text)
    print(f"\t‣ Bash Script Saved to {file_path}")
    job_id = None
    successful_sub_flag = False
    result = subprocess.run([RUN_COMMAND, file_path], capture_output=True, text=True)
    if result.returncode == 0:
        print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
        successful_sub_flag = False
        job_id = None
        
def migrateIslands(islands):
    # Load checkpoint data for every island
    # add some individuals to other islands
    # Save them back to checkpoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('checkpoints', type=str, help='Save Dir')
    parser.add_argument('islands', type=int, help='Number of Islands', default=2)
    # Parse the arguments
    args = parser.parse_args()
    temp_file = "temptemptemp.sh"
    generations = 10
    
    for gen in range(generations) :
        for i in range(args.islands):
            checkpoint_path = os.path.join("checkpoints", "island_" + str(i))
            
            submit_job(temp_file, PYTHON_BASH_SCRIPT_TEMPLATE.format(i, checkpoint_path))
            
        if gen % 5 == 0:
            migrateIslands(args.islands)