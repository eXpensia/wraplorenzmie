import pylorenzmie
from pylorenzmie.theory import Instrument
from pylorenzmie.theory.Feature import Feature
from wrapper.utilities.utilities import normalize
from wrapper.utilities.utilities import crop
from tqdm import tqdm


class fitting(object):
    """ All the wrap aroud the pylorenzmie toolbox
    available at the grier lab github
    https://github.com/davidgrier/pylorenzmie """

    def __init__(
        image, wavelength, magnification, n_m=1.33, dark_count=0, background=1
    ):
        self.image = image
        self.shape = image.shape
        self.instrument = Instrument.Instrument(
            wavelength=wavelength,
            magnification=magnification,  # um/pixel
            n_m=n_m,
            dark_count=dark_count,
            background=background,
        )

    def make_guess(self, a_p, n_p, z, alpha=0.8, fit_nr=True, fit_alpha=False):
        """ Add the guess to the object to fit more nicely, z is to be put in
        µm for easier usage, it will be put in pixels in the actual guess dict
        !!! You need to give it an already cropped image."""
        if fit_nr == True:
            vary = [True] * 5
            vary.extend([False] * 5)
            self.vary = vary
        else:
            vary = [True] * 3
            self.vary = vary
            vary.extend([False] * 7)

        if fit_alpha == True:
            sef.vary[-1] = True

        self.shape = cropped_norm.shape
        self.guesses = {
            "a_p": a_p,  # um
            "n_p": n_p,
            "r_p": [self.shape[0] // 2, self.shape[1] // 2, z / self.magnification],
            "alpha": alpha,
        }

    def fit_single(self, image, method="lm"):
        """Fit an hologram with the given guesses"""

        fitter = Feature(**self.guesses)

        fitter.FitSettings.parameters.vary = self.vary
        fitter.model.instrument = instrument
        fitter.model.coordinates = Instrument.coordinates(shape)
        fitter.data = (image).reshape(image.size)
        return fitter.optimize(method=method)

    def update_guess(self, x, y, z):
        self.guesses["r_p"] = [x, y, z]

    def fit_video(self, vid, background, savefile, box_size, n_start=1, n_end=None):
        """Fit a full movie by just using the guesses of the first image
        the guesses for the next image will take the precedent ones.
        Box_size is the fitted square."""
        if n_end == None:
            print("Computing the length of the video")
            n_end = vid.get_length
            print("length of video = {}".format(self.number))
            data = np.zeros((n_end - n_start,))
        image = vid.get_image
        fp = np.memmap(
            savefile, dtype="float32", mode="w+", shape=(int(n_end - n_start), 6)
        )
        for n, i in enumerate(tqdm(range(n_start, n_end))):
            if i > n_start:
                image = normalize(vid.get_next_image(), background)

            image = crop(image, *guesses["r_p"][0:2], box_size)
            result = self.fit_single(image)
            fp[n, :] = np.append(result.result["x"], result.final["alpha"])
            # save result
            self.update_guess(*result.result["x"][0:3])
        del fp
