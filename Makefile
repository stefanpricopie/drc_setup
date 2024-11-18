history:
	git log --graph --decorate --oneline --all

move:
	mv results/*.pkl results/raw/

download:
	# Download results
	rsync -avz csf:"~/FinalERC/results/*" ./results/

upload:
	rsync -avz 1-launch.py csf:"~/FinalERC/"

JOB_ID= # Leave JOB_ID empty; it will be provided at runtime
TASK_ID= # Leave TASK_ID empty; it will be provided at runtime

view-task-mem:
	ifndef JOB_ID
		$(error JOB_ID is not set. Use make view-task-mem JOB_ID=<job-id> TASK_ID=<task-id>)
	endif
	ifndef TASK_ID
		$(error TASK_ID is not set. Use make view-task-mem JOB_ID=<job-id> TASK_ID=<task-id>)
	endif
		@qacct -j $(JOB_ID) | awk -v taskID=$(TASK_ID) '/jobnumber/ {jobnumber=$2} /taskid/ && $$2 == taskID {found=1} /maxvmem/ && found {print "JobNumber:", jobnumber, "TaskID:", taskID, "MaxVMem:", $$2; found=0;}'

# Use the make command with: make convert NOTEBOOK=my_notebook.ipynb
convert:
	jupyter nbconvert --to script $(NOTEBOOK)