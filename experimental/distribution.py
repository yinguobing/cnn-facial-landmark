"""Draw the histgram of the pose distributions

Run it like this:
    `python3 -m experimental.distribution.py`

Do not forget to set the dataset file path.
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_parsed_dataset
from experimental.pose_estimator import PoseEstimator

if __name__ == "__main__":

    ds = get_parsed_dataset("data/helen.record", 1, False)

    # Counters
    n_faces = 0
    pitches = []
    yaws = []
    rolls = []

    for image, marks in ds:
        # image = (image.numpy()[0]*255).astype(np.uint8)
        height, width = image.shape[1:3]
        pose_estimator = PoseEstimator(img_size=(height, width))
        marks = np.reshape(marks, (-1, 2))*width
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # Solve the pitch, yaw and roll angels.
        r_mat, _ = cv2.Rodrigues(pose[0])
        p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
        _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
        pitch, yaw, roll = u_angle.flatten()

        # I do not know why the roll axis seems flipted 180 degree. Manually by pass
        # this issue.
        if roll > 0:
            roll = 180-roll
        elif roll < 0:
            roll = -(180 + roll)

        pitches.append(pitch)
        yaws.append(yaw)
        rolls.append(roll)
        n_faces += 1

        # print("pitch: {:.2f}, yaw: {:.2f}, roll: {:.2f}".format(
        #     pitch, yaw, roll))

        # for mark in marks:
        #     cv2.circle(image, tuple(mark), 1, (0, 255, 0), 1)
        # cv2.imshow("image", image)
        # if cv2.waitKey() == 27:
        #     break

    fig, ax = plt.subplots(3, 1)
    ax[0].hist(pitches, 40, (-60, 60), density=True)
    ax[1].hist(yaws, 40, (-60, 60), density=True)
    ax[2].hist(rolls, 40, (-60, 60), density=True)
    
    plt.show()
    print(n_faces)
