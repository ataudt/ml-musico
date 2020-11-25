# ML-Musico <!-- omit in toc -->

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
  - [Settings](#settings)
  - [Start the programe](#start-the-programe)
  - [End the programe](#end-the-programe)
- [Credits](#credits)
  - [Face recognition from the audience](#face-recognition-from-the-audience)

-----------------------------

## Background

ML-Musico, short for "Machine Learning - Musico" is a digital composer, developed for an art project. During a live performance with a jazz band, it acts as an intermediate between the band and the audience. It is able to recognize emotions from the audience, and gives instructions to the performers based on the audience's emotions.

Please find more information about this performance on this website https://dreiorangen.wordpress.com/filmabend/.


## Installation

- Download and install miniconda (download from internet).
- Open a terminal and execute the following commands:
  - Create a new environment for this project with `conda create -n musico python=3.7`
  - Activate the environment with `conda activate musico`.
  - Clone the project from git with `git clone https://github.com/ataudt/ml-musico.git` and change into the created (cloned) directory.
  - After entering the cloned directory, install this project with `pip install .`

## Usage

### Settings

All available settings can be found and changed in the file `settings.yaml`. Please see the comments there on how it works.
You can specify which webcam to use as parameter `use_webcam` (starting with 0).

### Start the programe

Open a terminal and change to the installation directory. Activate the conda environment with `conda activate musico`. 
Use `python run_musico.py --help` for a list of available runtime options.
Use `python run_musico.py` to run the program for performance. After a short initial loading phase where you can rearrange the windows as needed, you will need to press **Enter** in the console to resume the program.

### End the programe

You can always stop the program by clicking on the video feed window, and pressing `q`.
Otherwise, the program will stop automatically after a certain time (specified as parameter `max_minutes_song` in the `settings.yaml` file).

## Credits

### Face recognition from the audience

Face recognition is based on this project https://github.com/petercunha/Emotion.git and uses a WebCam Feed with OpenCV and Deep Learning.