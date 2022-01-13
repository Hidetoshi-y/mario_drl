#!/bin/bash

pip install pyqt5 || exit 1
pip install imageio || exit 1
pip install gym-super-mario-bros || exit 1
pip install vcopt || exit 1
pip install opencv-python || exit 1
pip install torch || exit 1
pip install tensorboard || exit 1

#ImportError: libGL.so.1: cannot open shared object file: No such file or directoryならば
#pip install opencv-python || exit 1

#or

#apt-get update && apt-get upgrade -y
#apt-get install -y libgl1-mesa-dev