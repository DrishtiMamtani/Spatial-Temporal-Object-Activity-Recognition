"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
from UCFdata import DataSet
from processor import process_image
from keras.models import load_model


def main(nb_images=176):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('data/checkpoints/inception.017-2.46.hdf5')  # replaced by your model name
    images = []
    for seq in range(1, nb_images+1):
        seq=format(seq,"04")
        images.append('./data/test/Typing/v_Typing_g08_c01-{}.jpg'.format(seq))
    print(images)

    for j in range(nb_images):
        print('-' * 80)

        sample = j
        print(sample)
        image = images[sample]
        j=j+1


        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1


if __name__ == '__main__':
    main()