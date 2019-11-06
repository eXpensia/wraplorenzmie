import imageio
from scipy.ndimage import gaussian_filter as gaussian_filter
import numpy as np
from tqdm import tqdm
import pylorenzmie.detection.circletransform as ct
import pylorenzmie.detection.h5video as h5
import pylorenzmie.detection.localize as localize
import trackpy as tp


class video_reader(object):
    def __init__(self, filename, number=None, background=None, codecs=None):
        self.filename = filename
        self.codecs = codecs
        if codecs == None:
            self.vid = imageio.get_reader(self.filename)
        else:
            self.vid = imageio.get_reader(self.filename, codecs)
        self.number = number
        self.background = background

    def close(self):
        self.vid.close()

    def get_image(self, n):
        """ Get the image n of the movie """
        return np.array(self.vid.get_data(n)[:, :, 1])

    def get_next_image(self):
        return np.array(self.vid.get_next_data()[:, :, 1])

    def get_filtered_image(self, n, sigma):
        """ Get the image n of the movie and apply a gaussian filter """
        return gaussian_filter(self.vid.get_data(n)[:, :, 1], sigma=sigma)

    def get_background(self, n):
        """ Compute the background over n images spread out on all the movie"""
        if self.number == None:
            print("needs to compute length of the video.")
            self.get_length()
            print("length of video = {}".format(self.number))

        image = self.get_image(1)
        size = (n, image.shape[0], image.shape[1])
        buf = np.empty(size, dtype=np.uint8)
        with tqdm(total=n) as pbar:
            for i, toc in enumerate(np.arange(0, self.number, self.number // (n - 1))):
                buf[i, :, :] = self.get_image(toc)
                pbar.update(1)

        self.background = np.median(buf, axis=0)
        return self.background

    def get_length(self):
        """Read the number of frame of vid, can be long with some format as mp4
        so we don't read it again if we already got it"""
        if self.number == None:
            self.number = self.vid.count_frames()
            return self.number
        else:
            return self.number


def center_find(
    image,
    n_fringes=28,
    trackpy_params={"diameter": 30, "minmass": None, "topn": None, "engine": "numba"},
):
    """
    Courtesy of https://github.com/davidgrier/pylorenzmie/
    Example wrapper that uses orientational alignment transform (OAT)
    and trackpy.locate to return features in an image.

    Scheme for using OAT method:
    1) Use circletransform.py to turn rings into blobs.
    2) Use trackpy.locate to locate these blobs.

    Keywords:
        trackpy_params: dictionary of keywords to feed into trackpy.locate.
                        See trackpy.locate's documentation to learn to use
                        these to optimize detection. These can be a bit tricky!
    Returns:
        features: matrix w/ columns ['xc', 'yc', 'w', 'h'] and rows as a
                  detection. (xc, yc) is the center of the particle, and
                  (w, h) = (201, 201) by default.
        circ: circle_transform of original image for plotting.
    """
    circ = ct.circletransform(image, theory="orientTrans")
    circ = circ / np.amax(circ)
    circ = h5.TagArray(circ, frame_no=None)
    features = tp.locate(circ, **trackpy_params)
    features["w"] = 201
    features["h"] = 201
    features = np.array(features[["x", "y", "w", "h"]])

    # Find extent of detected features and change default bounding box.
    for idx, feature in enumerate(features):
        xc = feature[0]
        yc = feature[1]
        s = localize.feature_extent(norm, (xc, yc), nfringes=nfringes)
        features[idx][2] = s
        features[idx][3] = s

    return features, circ


def plot_bounding(image, features):
    fig, ax = plt.subplots()
    ax.imshow(norm, cmap="gray")
    for feature in features:
        x, y, w, h = feature
        test_rect = Rectangle(
            xy=(x - w / 2, y - h / 2),
            width=w,
            height=h,
            fill=False,
            linewidth=3,
            edgecolor="r",
        )
        ax.add_patch(test_rect)
    plt.plot()


def normalize(image, background, dark_count=None):
    if dark_count == None:
        return (image - dark_count) / (background - dark_count)
    else:
        return image / backgroud


def crop(image, x, y, h):
    return image[y - h // 2 : y + h // 2, x - h // 2 : x + h // 2]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    t = time.time()
    vid = video_reader(
        "/media/maxime/Maxtor/09102019/film75fps1_15k/Basler_acA1920-155um__22392621__20191009_143652597.mp4"
    )
    vid.number = 156604
    bg = vid.get_background(100)
    t = time.time() - t
    vid.close()
    print(t)
    plt.imshow(bg, cmap="gray")
    plt.show()
