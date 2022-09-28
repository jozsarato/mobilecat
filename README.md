## MobTrackCat
Automatic fixation categorization for mobile eye tracking 

developed for the output of the pupil labs eye-tracker


package has two main functionalities.
from a mobile eye tracker videos:
1. cut frames around gaze location in a small rectangle (eg: 96*96 pixels) and export as images -- for training and test videos
training videos should contain only looking at a single stimulus

2. use images from the training videos to train a convoluational neural network with the keras library. 

3. use the above trained network to classify images from the test videos 

4. aggregate the classification using temporal information
