
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Load the dataset """
    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/GRIP_ForgeVideoDataset/"
    video_input = sorted(glob(os.path.join(dataset_path, "*", "forged.avi")))
    video_output = sorted(glob(os.path.join(dataset_path, "*", "mask.avi")))

    """ Create directory """
    create_dir("data")

    for x, y in tqdm(zip(video_input, video_output), total=len(video_output)):
        """ Extracting the name """
        x_name = x.split("/")[-2]
        y_name = y.split("/")[-2]

        """ Create directories """
        create_dir(f"data/{x_name}/image")
        create_dir(f"data/{x_name}/mask")

        """ Capturing videos """
        x_cap = cv2.VideoCapture(x)
        y_cap = cv2.VideoCapture(y)

        """ Loop over videos """
        index = 0
        while True:
            """ Reading frame from videos """
            x_ret, x_frame = x_cap.read()
            y_ret, y_frame = y_cap.read()

            """ Breaking the loop """
            if x_ret == False or y_ret == False:
                break

            try:
                """ Save the frames """
                y_frame = cv2.cvtColor(y_frame, cv2.COLOR_RGB2GRAY)
                y_frame = y_frame/255.0
                y_frame = y_frame > 0.5
                y_frame = y_frame.astype(np.int32)
                y_frame = y_frame * 255

                if 255 not in np.unique(y_frame):
                    pass
                else:
                    cv2.imwrite(f"data/{x_name}/image/{index}.png", x_frame)
                    cv2.imwrite(f"data/{x_name}/mask/{index}.png", y_frame)

                """ Incrementing  the index """
                index += 1

            except Exception as e:
                pass
