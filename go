#!/bin/bash

OUTDIR=_out
DEFAULT_TARGET=default
cd $(dirname $0)
BASEDIR=$(pwd)

function main {
    if [ "$1" == "" ]; then
	      TARGET=$DEFAULT_TARGET
    else
	      TARGET=$1
    fi

    $TARGET
}

function build {
    mkdir -p $OUTDIR
    cd $OUTDIR
    cmake .. && make && cd ..
}

function default {
    build
}

function ps1 {
    build
    cd $BASEDIR/ps1
    ../_out/ps1/main
}

function clean {
    rm -rf $OUTDIR
}

main $*
