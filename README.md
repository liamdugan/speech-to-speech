# Simultaneous Speech to Speech translation w/ Whisper
The following repository contains code for querying whisper in a simultaneous fashion

## Setup instructions
First clone the repo and cd into the main directory
``` 
git clone git@github.com:liamdugan/speech-to-speech.git 
cd speech-to-speech
```

Then (with conda installed) run
```
conda env create -f environment.yml -p ./env
conda activate ./env
```

To check to see if the installation worked run
```
python pipeline.py
```
