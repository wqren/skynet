#!/bin/bash

tarfile=$1
tarlocal=$(basename $tarfile)
zipfile=${tarlocal/.tar/.zip}
datadir=/tmp/imagenet-data-$tarlocal

echo Working... $tarlocal
(
  rm -rf $datadir &&
  mkdir -p $datadir &&
  cp $tarfile $datadir &&
  cd $datadir &&
  tar xf $tarlocal &&
  zip -q -0 $zipfile *.JPEG &&
  cp $zipfile /hdfs/imagenet-zip
)

rm -rf $datadir
