#!/usr/bin/env python3
import datetime
import os
import platform
import subprocess

completed_process = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True,
                                   stdout=subprocess.PIPE, universal_newlines=True)
# Strip newline character at the end
latest_git_hash = completed_process.stdout.strip()
if latest_git_hash is None:
    raise ValueError("Could not obtain the latest git hash")

# Assuming this remains constant as in your bash script
PARALLEL_ENV = "smp.pe"
NB_PARALLEL_PROCESS = ""  # You can specify this value as required
EMAIL = "stefan.pricopie@postgrad.manchester.ac.uk"
N_RUNS = 20


# Get the current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Assuming BINDIR remains constant as in your bash script
BINDIR = os.path.dirname(os.path.abspath(__file__))
# Modify OUTDIR to include both the current date and the latest git hash
OUTDIR = f"results/{current_date}_{latest_git_hash}"
# Ensure the directory exists
os.makedirs(OUTDIR, exist_ok=True)


def qsub_job(runner, configs, jobname):
    # Generate the Config array
    config_array = "configs=(" + " \\\n         \"" + "\" \\\n         \"".join(configs) + "\")"

    cmd = f"""#!/bin/bash --login
#$ -t 1-{len(configs)}  # Using N_RUNS to specify task range
#$ -N {jobname}
# -pe {PARALLEL_ENV} {NB_PARALLEL_PROCESS}
#$ -l s_rt=06:00:00
# -l mem512                   # For 32GB per core
# -M {EMAIL}
# -m as
#$ -cwd
#$ -j y
#$ -o {OUTDIR}

{config_array}

# Use SGE_TASK_ID to access the specific configuration
CONFIG_INDEX=$(($SGE_TASK_ID - 1))  # Arrays are 0-indexed
CONFIG=${{configs[$CONFIG_INDEX]}}

echo "{runner} $CONFIG"
echo "Job: $JOB_ID, Task: $SGE_TASK_ID, Config: $CONFIG"

{BINDIR}/{runner} $CONFIG
"""
    with subprocess.Popen(["qsub", "-v", "PATH"], stdin=subprocess.PIPE) as proc:
        proc.communicate(input=cmd.encode())


def add_config(configurations, problem, dim, algo, switching_cost, xcs, k=None, p=None, outdir=OUTDIR):
    begin_seed = 0
    for seed in range(begin_seed, begin_seed + N_RUNS):
        # Initialize base config string
        config = (f"--problem {problem} --dim {dim} --algo {algo} "
                       f"--switching_cost {switching_cost} --xcs {xcs} --seed {seed} --outdir {outdir}")

        # Add p or k to the config string based on which is provided
        if p is not None:
            config = f"{config} --p {p}"
        elif k is not None:
            config = f"{config} --k {k}"

        configurations.append(config)


def run_local(runner, configs):
    for i, config in enumerate(configs):
        cmd = [runner]
        cmd.extend(config.split())
        subprocess.run(cmd)


def run_job(job):
    runner = "run.py"       # Your Python script for running a single experiment
    configurations = job()  # Generate the configurations for the job

    if platform.system() == "Linux":
        # assert N_RUNS == 50, "N_RUNS must be 50 for cluster runs"
        # split configurations into jobnames and configs
        qsub_job(runner=runner, configs=configurations, jobname=job.__name__)
    elif platform.system() == "Darwin":  # macOS is identified as 'Darwin'
        assert N_RUNS == 1, "N_RUNS must be 1 for local runs"
        run_local(runner=f"{os.getcwd()}/{runner}", configs=configurations)


def ps_pfbo():
    configurations = []

    problems = [
        'ackley',
        'griewank',
        'levy',
        'michalewicz',
        'rosenbrock',
        'salomon',
        'schwefel',
    ]

    switching_cost = 15

    for dim in [2, 3, 4]:
        for xcs in range(1, dim):
            for problem in problems:
                for p in range(0, 101, 5):
                    p /= 100
                    add_config(configurations, problem=problem, dim=dim, algo="bo_random",
                                switching_cost=switching_cost, xcs=xcs, p=p)

                for k in [1, 2, 4, 8, 16]:
                    add_config(configurations, problem=problem, dim=dim, algo="pbo", switching_cost=switching_cost, xcs=xcs, k=k)

    return configurations


def run_final():
    configurations = []

    problems = [
        'ackley',
        'griewank',
        'levy',
        'michalewicz',
        'rosenbrock',
        'salomon',
        'schwefel',
    ]

    switching_cost = 31

    for dim in [4]:
        for xcs in [1]:
            for problem in problems:
                add_config(configurations, problem=problem, dim=dim, algo="bo", switching_cost=switching_cost, xcs=xcs)

                add_config(configurations, problem=problem, dim=dim, algo="bo_random", switching_cost=switching_cost, xcs=xcs, p=0.25)

                for k in [2, 3, 4, 5]:
                    add_config(configurations, problem=problem, dim=dim, algo="bo_random", switching_cost=switching_cost, xcs=xcs, p=1-1/k)
                    add_config(configurations, problem=problem, dim=dim, algo="pbo", switching_cost=switching_cost, xcs=xcs, k=k)
                    add_config(configurations, problem=problem, dim=dim, algo="pbonested", switching_cost=switching_cost, xcs=xcs, k=k)

                for sc in [1, 3, 7, 15, 31]:
                    add_config(configurations, problem=problem, dim=dim, algo="eipu", switching_cost=sc, xcs=xcs)

    return configurations


if __name__ == "__main__":
    jobs = [
        # ps_pfbo,
        run_final,
    ]
    for job in jobs:
        run_job(job)
