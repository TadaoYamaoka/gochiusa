#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make an object
#   detector for things like faces, pedestrians, and any other semi-rigid
#   object.  In particular, we go though the steps to train the kind of sliding
#   window object detector first published by Dalal and Triggs in 2005 in the
#   paper Histograms of Oriented Gradients for Human Detection.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import os
import sys
import glob

import dlib
from skimage import io


# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.
if len(sys.argv) != 3:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]
xml_file = sys.argv[2]


# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True


training_xml_path = os.path.join(faces_folder, xml_file)
# This function does the actual training.  It will save the final detector to
# detector.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)



# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
