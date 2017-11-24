#!/bin/bash

VNCPORT=$((5910+$1))

vncviewer localhost:$VNCPORT

