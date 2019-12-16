CNN (Inception V3) is used to make the model.
 The model is loaded in frames_for_video.py.
  It then takes a video as input(new1), converts it into frames, predicts confidence scores for each frame and displays it as a plot.
  The frames are simultaneously deleted to reduce space consumption.
  For a bundle of 10 frames, only one activity which has the most probability is predicted.
 