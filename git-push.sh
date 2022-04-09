#!/bin/bash

dating=`date`

git status
git add . -A
git commit -m "$dating $1"
git push origin main
git log 
