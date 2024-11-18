#!/bin/bash
set -e
set -o pipefail

# Find our own location.
BINDIR=$(dirname "$(readlink -f "$(type -P $0 || echo $0)")")
OUTDIR="./stdout/"


# Function to run a local job
local_job() {
    RUNNER=$1
    RUN_ID=$2
    KEY=$3
    EPOCH=$4
    ORDER=$5
    STRATEGY=$6
    RHO=$7
    echo "Running ${BINDIR}/${RUNNER} Run ID ${RUN_ID}, Key ${KEY}, Epoch ${EPOCH}, Order ${ORDER}, Strategy ${STRATEGY}, Rho ${RHO}"
    ${BINDIR}/${RUNNER} ${RUN_ID} ${KEY} ${EPOCH} ${ORDER} ${STRATEGY} ${RHO} > ${OUTDIR}/${RUNNER}_${RUN_ID}_${KEY}_${EPOCH}_${ORDER}_${STRATEGY}_${RHO}.stdout 2>&1
}

# Parallelize based on nruns
export -f local_job
export BINDIR
export OUTDIR

# test
nruns=3  # Setting to 3
parallel --jobs ${nruns} "local_job run.py {1} {2} {3} {4} {5}" ::: $(seq 1 ${nruns}) ::: ackley2d ::: 1 30 ::: 0 2 ::: waiting commitment
#parallel --jobs ${nruns} "local_job run.py {1} {2} {3} {4} {5}" ::: $(seq 1 ${nruns}) ::: ackley3d styblinski3d ::: 1 30 ::: 0 3 S111 ::: waiting commitment dynwaiting

# run
# nruns=30  # Setting to 30

# Run 2d
# parallel --jobs ${nruns} "local_job run.py {1} {2} {3} {4} {5}" ::: $(seq 1 ${nruns}) ::: eggholder ::: 1 5 10 15 20 25 30 ::: 0 1 2 S11 SN1 ::: waiting commitment dynwaiting
# Run 3d
#parallel --jobs ${nruns} "local_job run.py {1} {2} {3} {4} {5}" ::: $(seq 1 ${nruns}) ::: ackley3d styblinski3d ::: 1 5 10 15 20 25 30 ::: 0 1 2 3 S111 ::: waiting commitment dynwaiting

# discrete
# nruns=30  # Setting to 30
# parallel --jobs ${nruns} "local_job run.py {1} {2} {3} {4} {5}" ::: $(seq 1 ${nruns}) ::: onemin30d ::: 1 5 10 15 20 25 30 ::: 0 1 2 3 4 5 6 ::: waiting commitment dynwaiting
