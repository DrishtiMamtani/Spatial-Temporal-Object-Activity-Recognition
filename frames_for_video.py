#The code generates frames and predicts confidence score for frames
import cv2
import os
import time

from PIL import Image
import numpy as np
import operator
import random
import glob
from UCFdata import DataSet
from processor import process_image
from keras.models import load_model
import matplotlib.pyplot as plt
def video(video_name):#generating frames
    cap = cv2.VideoCapture(video_name)

    while True:

        ret, frame = cap.read()

        if ret == True:

            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        else:
            break



def video_to_frames(video_name):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index

    vidcap = cv2.VideoCapture(video_name)
    count = 0
    num=0
    skip=-1
    anslst=[]
    acclst=[]
    data = DataSet()
    model = load_model('data/checkpoints/inception.017-2.46.hdf5')  # replaced by your model name
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(os.getcwd(), '%d.png') % num, image)
            flag=1
            while(flag is 1):
                if skip % 10 == 0:#display output for 10 images together
                    print('-' * 80)
                skip=skip+1

                images = '{}.png'.format(num)
                if skip % 10 == 0:
                    print(images)#output as 10,20,30.png

                image_arr = process_image(images, (299, 299, 3))
                image_arr = np.expand_dims(image_arr, axis=0)

                # Predict.
                predictions = model.predict(image_arr)

                # Show how much we think it's each one.
                label_predictions = {}
                for i, label in enumerate(data.classes):
                    label_predictions[label] = predictions[0][i]

                sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
                ans=sorted_lps[0][0]#top activity
                acc=sorted_lps[0][1]#percentage of top activity
                if acc>0.4:#threshold is 40 percent
                    anslst.append(ans)#add the first confidence score to the list only when it is more than 40 percent
                if skip%10==0:
                    print(max(anslst,key=anslst.count))#print the most frequent activity
                    anslst=['0']#clear the list

                im = Image.open(images)
                plt.imshow(im)#plot frames
                plt.title(ans)#show the top activity as the title
                plt.show()

                plt.clf()  # will make the plot window empty
                im.close()
                os.remove(images)#remove images
                # time.sleep(0.001)

                num=num+1
                count += 1

                break


        else:
            flag=0
            num=num+1
            count=count+1
            continue
    cv2.destroyAllWindows()
    vidcap.release()
    print("end")

#video('result.mp4')
print("start")
video_to_frames('new1.mp4')#new1 should be in the same folder as this code















