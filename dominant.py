# IMPROVEMENTS
# This code updates the histogram rather than showing a static image
# It also samples the colors from a rectangle in the screen

# REFERENCES
# Finding dominant color https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# Updating histogram https://www.geeksforgeeks.org/how-to-update-a-plot-in-matplotlib/

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
    
cap = cv2.VideoCapture(0)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

while(1):
    # Take each frame
    _, frame = cap.read()

    # create rectangle at center of screen
    x = 500
    y = 200
    w= 400
    h = 400

    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # find dominnat color
    img = frame[y:y+h, x:x+w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    # plot histogram
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    plt.axis("off")
    plt.imshow(bar)

    # update histogram
    fig.canvas.draw()
    fig.canvas.flush_events()

cv2.destroyAllWindows()



