#!/bin/bash

VNCPORT=$((7000+$1))

vncviewer localhost:$VNCPORT

