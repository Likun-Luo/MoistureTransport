#!/bin/bash

PROGRAM="MoistureTransport Simulation"
VERSION="0.1"

declare FLAGS=( "--standalone"
                "--plugin-enable=numpy"
                "--plugin-enable=pylint-warnings")
PYFLAGS="-O"
BUILD_DIR="build/"
PYCOMPILER="python -m nuitka"
# CCOMPILER= --clang
# JOBS=-j 4
MAIN="main.py"

CALL="$PYCOMPILER ${FLAGS[@]} --python-flag=$PYFLAGS --output-dir=$BUILD_DIR $MAIN"

### Build execution

echo "Starting build of $PROGRAM:v$VERSION"
SECONDS=0
START=$(date)
echo "Starting at: $START"
echo ""
echo "##########################################################"

echo $CALL
$CALL
#sleep 3

DURATION=$SECONDS
STOP=$(date)
echo ""
echo "##########################################################"
echo "Build finished: $STOP"
echo "Build time: $(($DURATION / 60))min $(($DURATION % 60))sec"
#echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
