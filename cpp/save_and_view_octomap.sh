#!/bin/bash

mkdir mapfiles
FILENAME=`tempfile -d mapfiles -p map_ -s .ot`
rosrun octomap_server_ext octomap_saver -f $FILENAME
octovis $FILENAME

