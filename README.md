# Translatotron 2
Code for our re-implementation of translatotron 2

## Setup instructions
First clone the repo and cd into the main directory
``` 
git clone git@github.com:liamdugan/translatotron2.git 
cd translatotron2
```

Then (with conda installed) run
```
conda env create -f environment.yml -p ./env
conda activate ./env
```

To check to see if the installation worked run
```
python examples/transformer.py
```