import pylorenzmie
from pylorenzmie.theory import Instrument
from pylorenzmie.theory.Feature import Feature
from wraplorenzmie.utilities.utilities import normalize
from wraplorenzmie.utilities.utilities import crop
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class fitting(object):
    """ All the wrap aroud the pylorenzmie toolbox
    available at the grier lab github
    https://github.com/davidgrier/pylorenzmie """

    def __init__(
        self, image, wavelength, magnification, n_m=1.33, dark_count=0, background=1
    ):
        self.image = image
        self.magnification = magnification
        self.shape = image.shape
        self.instrument = Instrument.Instrument(
            wavelength=wavelength,
            magnification=magnification,  # um/pixel
            n_m=n_m,
            dark_count=dark_count,
            background=background,
        )

    def make_guess(self, a_p, n_p, z, alpha=1, fit_r=True, fit_n=True, fit_alpha=False):
        """ Add the guess to the object to fit more nicely, z is to be put in
        µm for easier usage, it will be put in pixels in the actual guess dict
        !!! You need to give it an already cropped image."""
        self.fit_n = fit_n
        self.fit_r = fit_r
        self.fit_alpha = fit_alpha
        self.shape = self.image.shape
        self.guesses = {
            "a_p": a_p,  # um
            "n_p": n_p,
            "r_p": [self.shape[0] // 2, self.shape[1] // 2, z / self.magnification],
            "alpha": alpha,
        }

    def set_vary(self, fitter):
        if self.fit_r:
            self.fitter.vary["a_p"] = True
        else:
            self.fitter.vary["a_p"] = False
        if self.fit_n:
            self.fitter.vary["n_p"] = True
        else:
            self.fitter.vary["n_p"] = False
        if self.fit_alpha:
            self.fitter.vary["alpha"] = True
        else:
            self.fitter.vary["alpha"] = False

    def _crop_fit(self, image):
        return np.array(crop(image, self.xc, self.yc, self.h))

    def fit_single(self, image, method="lm"):
        """Fit an hologram with the given guesses"""
        self.fitter = Feature(**self.guesses)
        self.set_vary(self.fitter)
        self.fitter.model.instrument = self.instrument
        try:
            self.fitter.model.instrument.background = np.mean(image)
        except:
            self.fitter.model.instrument.background = 1
        self.fitter.model.coordinates = Instrument.coordinates(self.shape)
        self.fitter.data = (image).reshape(image.size)
        return self.fitter.optimize(method=method)

    def update_guess(self, z):
        self.guesses["r_p"][2] = z

    def fit_video(self, xc, yc, vid, savefile, h, n_start=1, n_end=None, method="lm", dark_count_mode="min"):
        """Fit a full movie by just using the guesses of the first image
        the guesses for the next image will take the precedent ones.
        Box_size is the fitted square. To initialize correctly the fitter
        it's needed to give it the the position of the crop"""
        self.xc = int(xc)
        self.yc = int(yc)
        self.h = h

        if n_end == None:
            print("Computing the length of the video")
            n_end = vid.get_length
            print("length of video = {}".format(self.number))

        image = vid.get_image(1)
        _crop_fit = self._crop_fit
        if dark_count_mode ==  "min":
            image = normalize(_crop_fit(image), _crop_fit(vid.background), dark_count=np.min(_crop_fit(image)))
        elif dark_count_mode == "zero":
            image = normalize(_crop_fit(image), _crop_fit(vid.background), dark_count=0)
        elif dark_count_mode == "set":
            image = normalize(_crop_fit(image), _crop_fit(vid.background), dark_count=vid.dark_count)


        image = image / np.mean(image)
        self.fp = np.memmap(
            savefile, dtype="float64", mode="w+", shape=(int(n_end - n_start), 10)
        )
        for n, i in enumerate(tqdm(range(n_start, n_end))):
            if i > n_start:
                image = normalize(vid.get_next_image(), vid.background)
                image = self._crop_fit(image)
                try:
                    image = image / np.mean(image)
                except:
                    image = image

            self.result = self.fit_single(image, method=method)
            self._globalize_result()
            self.save_result(n)
            self.update_guess(self.result.final["z_p"])
        del self.fp

    def show_results(self):
        fit = self.fitter.model.hologram().reshape(self.shape)
        noise = self.fitter.noise
        mega_image = np.hstack([self.image, fit])
        fig, ax = plt.subplots(figsize=(12, 36))
        ax.imshow(mega_image, cmap="gray", interpolation=None)

    def _globalize_result(self):
        self.result.final["x_p"] = self.result.final["x_p"] + self.xc - self.h // 2
        self.result.final["y_p"] = self.result.final["y_p"] + self.yc - self.h // 2
        self.xc = int(self.result.final["x_p"])
        self.yc = int(self.result.final["y_p"])

    def save_result(self, n):
        buf = np.array([])
        for i in self.result.final.values():
            buf = np.append(buf, i)
        self.fp[n, :] = buf


def globalize_result(result, xc, yc, h):
    result.final["x_p"] = result.final["x_p"] + xc - h // 2
    result.final["y_p"] = result.final["y_p"] + yc - h // 2
    return result
