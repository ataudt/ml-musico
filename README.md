# ML-Musico

This art project develops a Machine counterpart for a musical live performance with a jazz trio.
It is able to recognize emotions from the audience, and gives instructions to the performers in order to achieve a predefined dramaturgy.

## Face recognition from the audience

Face recognition is based on this project https://github.com/petercunha/Emotion.git and uses a WebCam Feed with OpenCV and Deep Learning.

## Installation
    - Download and install miniconda (download from internet).
    - Open a terminal and create a new environment for this project with `conda create -n musico`
    - Activate the environment with `conda activate musico`.
    - Clone the project from git with `git clone https://github.com/ataudt/ml-musico.git` and change into the created (cloned) directory.
    - After entering the cloned directory, install this project with `pip install .`

## TODO:
    - timing in addition to frame info, done
    - pyplot for instructions, done
    - config file to specify instructions, done
    - implement events