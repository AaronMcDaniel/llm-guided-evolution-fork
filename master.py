import os
import argparse
import subprocess
import time

PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=LLMTest_Island_{}
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00
#SBATCH -oReport_islands-%j.out
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


# Run Python script
python islandIntegration.py {}
"""

def submit_run(tempFile, text):
    with open(tempFile, 'w') as file:
        file.write(text)
    print(f"\t‣ Bash Script Saved to {tempFile}")
    job_id = None
    successful_sub_flag = False
    result = subprocess.run(["sbatch", tempFile], capture_output=True, text=True)
    if result.returncode == 0:
        print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
        successful_sub_flag = False
        job_id = None
    
    return job_id
    
def check_contents_for_error(contents):
    """
    Checks the output of a job for any signs of error.

    Parameters:
    contents (str): output of job to check for error

    Returns:
    bool: True if job completed successfully, False if error, None if neither.  
    """

    # Check for error indicators in the file
    if "traceback" in contents.lower() or "slurmstepd: error" in contents.lower():
        print("\t☠ Error Found in LLM Job Output.", flush=True)
        return False
    elif "finished one generation" in contents.lower():
        print("\t☑ LLM Job Completed Successfully.", flush=True)
        return True
    else:
        return None

def check4job_completion(job_id, local_output=None, check_interval=60, timeout=3600*3):
    """
    Check for the completion of a job by searching for its output file and scanning for errors.

    Parameters:
    job_id (str): The job ID to check.
    check_interval (int): Time in seconds between checks.
    timeout (int): Maximum time in seconds to wait for job completion.

    Returns:
    bool: True if job completed successfully, False otherwise.
    """

    if local_output is not None:
        state = check_contents_for_error(local_output)
        if state is None:
            raise Exception('Unexpected output from job')
        else:
            return state

    start_time = time.time()
    output_file = f'Report_islands-{job_id}.out'

    while True:
        # Check if the timeout is reached
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for job completion.")
            return False

        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                state = check_contents_for_error(contents)
                if state is None:
                    pass
                else:
                    return state

        # Wait for some time before checking again
        time.sleep(check_interval)
        print(f'\t‣ Waiting on check4job_completion LLM job: {job_id} Time: {round(time.time() - start_time)}s', flush=True)

def migrateIslands(islands):
    # Load checkpoint data for every island
    # add some individuals to other islands
    # Save them back to checkpoints
    print("-" * 20)
    print("Simulating Migration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('checkpoints', type=str, help='Save Dir')
    parser.add_argument('--islands', type=int, help='Number of Islands', default=2)
    # Parse the arguments
    args = parser.parse_args()
    temp_file = "temptemptemp.sh"
    generations = 10
    
    for gen in range(generations) :
        print("Starting generation " + str(gen), flush=True)
        job_ids = []
        for i in range(args.islands):
            print("Generating Island " + str(i), flush=True)
            checkpoint_path = os.path.join("checkpoints", "island_" + str(i))
            
            job_id = submit_run(temp_file, PYTHON_BASH_SCRIPT_TEMPLATE.format(i, checkpoint_path))
            job_ids.append(job_id)
        
        done = True
        for i in range(len(job_ids)):
            done = check4job_completion(job_ids[i])
            if not done:
                break
        
        if not done:
            print("Error occured in loop, job not done")
            break
            
        if gen % 5 == 0:
            migrateIslands(args.islands)
    
    print("Finished evolutionary loop")