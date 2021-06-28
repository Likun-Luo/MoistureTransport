#!/bin/bash

OS=$(uname)
case $OS in 
    'Darwin')
        echo "MacOS"
        OS="MacOS"
        ;;
    'WindowsNT')
        echo "Windows"
        OS="Windows"
        ;;
    *)
        echo "Linux"
        OS="Linux"
        ;;
esac

VERSION="$(cat VERSION)"
PROGRAM="moisture_transport"
RELEASE_NAME="${PROGRAM}_v${VERSION}"
ONE_DIR=0
ICON="./icons/favicon/b074e2c6121e95222f48af881040a0a2.ico/android-icon-36x36.png"
RELEASE_DIR="release/$OS/$RELEASE_NAME"
WORK_DIR="./build/$PROGRAM.build"
DIST_DIR="./build/$PROGRAM.dist"


declare FLAGS=("--noconfirm"
                "--console" 
                "--icon $ICON"
                "--distpath $DIST_DIR"
                "--workpath $WORK_DIR"
                "--name $PROGRAM" 
                "--log-level WARN"
                "--name $PROGRAM")
if [[ $ONE_DIR == 0 ]]; then
    FLAGS+=("--onefile")
else
    FLAGS+=("--onedir"
            "--add-data ./cfg:cfg/")
fi

BUILD_DIR="release/"
PYCOMPILER="pyinstaller"
MAIN="main.py"

CALL="$PYCOMPILER ${FLAGS[@]} $MAIN"

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

echo "Moving executable release directory and preparing release package..."

#mv "$DIST_DIR/$APP_NAME"
echo $RELEASE_DIR
mkdir -pv $RELEASE_DIR
mkdir -pv $RELEASE_DIR/results

cp $DIST_DIR/$PROGRAM $RELEASE_DIR/.
cp -r cfg $RELEASE_DIR/cfg
cp README.md $RELEASE_DIR/.
