#!/bin/bash

message=""

if [ "$1" = "" ]
then message="@hyez is too busy to write commit message."
else message=$1
fi

git add .
git commit -m "$message"
git push -u origin master