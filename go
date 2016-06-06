#!/bin/bash

OUTDIR=_out
DEFAULT_TARGET=default
BASEDIR=$(dirname $0)

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

function run {
    build
    ./_out/src/main
}

function test {
    build
    cd ./_out
    ctest --output-on-failure
}

function clean {
    rm -rf $OUTDIR
}

cd $BASEDIR
main $*