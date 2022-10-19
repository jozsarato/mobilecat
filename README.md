## mobilecat

Automatic fixation categorization for mobile eye tracking  using deep learning

requires training videos, recorded with the mobile eye-tracker for each stimulus category of interest
developed for the output of the pupil labs eye-tracker


From a mobile eye tracker videos, the following tools are implemented:

1. cut frames around gaze location in a small rectangle (eg: 96*96 pixels) and export as images -- for training and test videos
training videos should contain only looking at a single stimulus  (preprocessing.py)

2. use images from the training videos to train a convoluational neural network with the keras library. (classmodels.py)
-- optional training with the data augmentation-- or random selection of traning images at each epoch (equal for each category).
-- to load images for training, each stimulus category should within a subfolder of a main folder (path of main folder should be provided)

3. use the above trained network to classify images from the test videos  (classmodels.py)
--predicitons are made for each sample of each frame of the test video
-- visualize prediction time-series & visualize predicitons for some example images

4. aggregate the classification using temporal information  (.py file in development)
-- use a dynamic model, for a best temporal prediction (discrete hidden markov model ?)


## installation instructions:
1. clone repository
2. in terminal/anaconda prompt, navigate to root folder of repository
3. execute: pip install -e .      (editable installation)  

###  dependencies: 

pandas

numpy

tensorflow    # the above installs it, if not previously installed

scikit-image  # the above installs it, if not previously installed 

opencv  (cv2)   # has to be installed separetely, (naming different in conda and pip, cv2, opencv, opencv-python) 
