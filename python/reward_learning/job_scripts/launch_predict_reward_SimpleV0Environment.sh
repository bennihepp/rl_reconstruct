#!/bin/bash

USERNAME="t-behepp"
read -s -p "Philly password for user ${USERNAME}: " PASSWORD
echo

CLUSTER="rr1"
WRAPPERSCRIPT="run_job_wrapper_philly.sh"
SPECIAL_NAME="test_abcd"
VC="intvc"
NUM_GPUS="2"
DOCKER_NAME="custom-py2-7-12-tf-1-2-1"
ONE_PROCESS_PER_CONTAINER="true"
# ONE_PROCESS_PER_CONTAINER="false"

SINGLE_CONTAINER_JOB="true"
DYNAMIC_CONTAINER_SIZE="false"
NUM_OF_CONTAINERS="1"

DEBUGGING="false"
# DEBUGGING="true"

if [ "$DEBUGGING" == "true" ]; then
	read -p "Launch jobs in debugging mode? [y/n]: " CONFIRM_YN
	if [ "$CONFIRM_YN" != "y" ]; then
		echo "Not launching debugging jobs"
		exit 1
	fi
	echo "Launching debugging jobs"
fi

for i in {85..85}; do
	JOBSCRIPT=`printf "run_job_%04d.sh" $i`
	if [ "$DEBUGGING" == "true" ]; then
		JOBNAME="debug_${JOBSCRIPT}"
	else
		JOBNAME="${JOBSCRIPT}_${SPECIAL_NAME}"
	fi

	CMD="https://philly/api/submit?"
	CMD+="buildId=0000&"
	CMD+="customDockerName=${DOCKER_NAME}&"
	CMD+="toolType=cust&"
	CMD+="clusterId=${CLUSTER}&"
	CMD+="vcId=${VC}&"
	CMD+="configFile=${USERNAME}%2Frl_reconstruct%2Fpython%2Freward_learning%2Fjob_scripts%2F${WRAPPERSCRIPT}&"
	CMD+="minGPUs=${NUM_GPUS}&"
	CMD+="name=${JOBNAME}&"
	CMD+="isdebug=${DEBUGGING}&"
	CMD+="iscrossrack=false&"

    # Data directory
	# CMD+="inputDir=%2Fhdfs%2F$VC%2F${USERNAME}%2Freward_learning%2Fdatasets%2F16x16x16_0-1-2-3_SimpleV0Environment_V2%2F&"
	CMD+="inputDir=%2Fhdfs%2F$VC%2F${USERNAME}%2Freward_learning%2Fdatasets%2Fin_out_grid_16x16x16_0-1-2-3_SimpleV0Environment_greedy%2F&"

	CMD+="extraParams=${JOBSCRIPT}&"
	CMD+="oneProcessPerContainer=${ONE_PROCESS_PER_CONTAINER}&"
	if [ "${SINGLE_CONTAINER_JOB}" == "true" ]; then
		CMD+="dynamicContainerSize=${DYNAMIC_CONTAINER_SIZE}&"
		CMD+="numOfContainers=${NUM_OF_CONTAINERS}&"
	fi
	CMD+="userName=${USERNAME}"

	echo "$CMD"

	curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"
done
