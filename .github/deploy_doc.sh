#!/bin/bash
set -ex

function deploy_doc(){
	echo "Creating doc at commit $1 and pushing to folder $2"
	git checkout $1
	cd "$GITHUB_WORKSPACE"
	pip install -U .
	cd "$GITHUB_WORKSPACE/docs"
	if [ ! -z "$2" ]
	then
		if [ "$2" == "main" ]; then
		    echo "Pushing main"
			make clean && make html && scp -r -oStrictHostKeyChecking=no _build/html/* $DOC_HOST:$DOC_PATH/$2/
			cp -r _build/html/_static .
		elif ssh -oStrictHostKeyChecking=no $DOC_HOST "[ -d $DOC_PATH/$2 ]"; then
			echo "Directory" $2 "already exists"
			scp -r -oStrictHostKeyChecking=no _static/* $DOC_HOST:$DOC_PATH/$2/_static/
		else
			echo "Pushing version" $2
			make clean && make html
			rm -rf _build/html/_static
			cp -r _static _build/html
			scp -r -oStrictHostKeyChecking=no _build/html $DOC_HOST:$DOC_PATH/$2
		fi
	else
		echo "Pushing stable"
		make clean && make html
		rm -rf _build/html/_static
		cp -r _static _build/html
		scp -r -oStrictHostKeyChecking=no _build/html/* $DOC_HOST:$DOC_PATH
	fi
}


# You can find the commit for each tag on https://github.com/huggingface/accelerate/tags
deploy_doc "main" main
deploy_doc "0fbbbc5" # v0.1.0 Latest stable release