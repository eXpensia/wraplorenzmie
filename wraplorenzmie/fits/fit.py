import wraplorenzmie.pylorenzmie
from wraplorenzmie.pylorenzmie.theory import Instrument
from wraplorenzmie.utilities.utilities import normalize
from wraplorenzmie.utilities.utilities import crop
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from wraplorenzmie.pylorenzmie.analysis import Feature
from wraplorenzmie.pylorenzmie.theory import LMHologram
from wraplorenzmie.pylorenzmie.utilities import coordinates
from wraplorenzmie.pylorenzmie.utilities import azistd
import matplotlib as mpl

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.markeredgecolor"] = "k"
mpl.rcParams["lines.markeredgewidth"] = 0.1
mpl.rcParams["figure.dpi"] = 130
from matplotlib import rc

rc("font", family="serif")
rc("text", usetex=False)
rc("xtick", labelsize="medium")
rc("ytick", labelsize="medium")


def cm2inch(value):
    return value / 2.54


class fitting(object):
    """All the wrap aroud the pylorenzmie toolbox
    available at the grier lab github
    https://github.com/davidgrier/pylorenzmie"""

    def __init__(
        self,
        image,
        wavelength,
        magnification,
        n_m=1.33,
        dark_count=0,
        background=1,
        double_precision=False,
        model=LMHologram,
        mask="fast",
        percentpix=0.1,
    ):
        if mask not in ["uniform", "radial", "donut", "fast"]:
            raise ValueError(
                "Mask should be one of the following : uniform, radial, donut or fast"
            )
        self.image = image
        self.magnification = magnification
        self.shape = image.shape

        self.feature = Feature(model=model(double_precision=double_precision))
        self.feature.data = image / np.mean(image)
        self.feature.coordinates = coordinates(image.shape)
        self.feature.mask.percentpix = percentpix
        self.feature.mask.distribution = mask
        self.ins = self.feature.model.instrument
        self.ins.wavelength = wavelength
        self.ins.magnification = magnification
        self.ins.n_m = n_m

    def show_mask(self):
        fig, (axa, axb) = plt.subplots(
            ncols=2,
            figsize=(5 * 1.2, 2.5 * 1.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axa.imshow(self.feature.mask.selected.reshape(self.image.shape))
        axa.axis("off")
        index = self.feature.mask.selected
        coords = self.feature.coordinates[:, index]
        axb.scatter(
            coords[0, :],
            coords[1, :],
            c=self.feature.data.flatten()[index],
            cmap="gray",
        )
        axb.axis("off")
        plt.show()

    def make_guess(
        self,
        a_p,
        n_p,
        z,
        r_p = None,
        alpha=1,
        show_estimate=False,
    ):
        """Add the guess to the object to fit more nicely, z is to be put in
        µm for easier usage, it will be put in pixels in the actual guess dict
        !!! You need to give it an already cropped image."""

        self.shape = self.image.shape

        # self.feature.properties["alpha"] = alpha
        # self.feature.properties.update(self.aberrations.properties)

        p = self.feature.particle
        if r_p == None:
            p.r_p = [self.shape[0] // 2, self.shape[1] // 2, z / self.ins.magnification]
        else:
            p.r_p = [r_p[0], r_p[1], z / self.ins.magnification]
        p.a_p = a_p
        p.n_p = n_p

        if show_estimate:
            fig, (axa, axb) = plt.subplots(
                ncols=2,
                figsize=(5 * 1.2, 2.5 * 1.2),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            axa.imshow(self.feature.data, cmap="gray")
            axb.imshow(self.feature.hologram(), cmap="gray")

            axa.axis("off")
            axa.set_title("Data")

            axb.axis("off")
            axb.set_title("Initial Estimate")

            plt.show()

    def set_vary(self, vary):
        parameters = [
            "k_p",
            "n_m",
            "alpha",
            "wavelength",
            "magnification",
            "piston",
            "xtilt",
            "ytilt",
            "defocus",
            "xastigmatism",
            "yastigmatism",
            "xcoma",
            "ycoma",
            "spherical",
            "x_p",
            "y_p",
            "z_p",
            "a_p",
            "n_p",
        ]
        fixed = [i for i in parameters if (i not in vary)]

        self.feature.optimizer.fixed = fixed

    def _crop_fit(self, image):
        return np.array(crop(image, self.xc, self.yc, self.h))

    def report(self, result):
        def value(val, err, dec=4):
            fmt = "{" + ":.{}f".format(dec) + "}"
            return (fmt + " +- " + fmt).format(val, err)

        keys = self.feature.optimizer.variables
        res = [i + " = " + value(result[i], result["d" + i]) for i in keys]

        print("npixels = {}".format(result.npix))
        print(*res, sep="\n")
        print("chisq = {:.2f}".format(result.redchi))

    def present(self, feature):
        fig, axes = plt.subplots(
            ncols=3, figsize=(5 * 1.2, 2.5 * 1.2), constrained_layout=True
        )

        vmin = np.min(feature.data) * 0.9
        vmax = np.max(feature.data) * 1.1
        style = dict(vmin=vmin, vmax=vmax, cmap="gray")

        images = [feature.data, feature.hologram(), feature.residuals() + 1]
        labels = ["Data", "Fit", "Residuals"]

        for ax, image, label in zip(axes, images, labels):
            ax.imshow(image, **style)
            ax.axis("off")
            ax.set_title(label)

        plt.show()

    def radial_profile(self, exp, theo, center):
        plt.figure(figsize=(5 * 1.2, 2.5 * 1.2))
        th_avg, expe_std = azistd(theo, center)
        rad_th = np.arange(len(th_avg)) * self.ins.magnification
        plt.plot(rad_th, th_avg, linewidth=2, label="Theory")

        expe_avg, expe_std = azistd(exp, center)
        rad_exp = np.arange(len(expe_avg)) * self.ins.magnification
        plt.plot(rad_exp, expe_avg, linewidth=2, label="Experimental")

        plt.fill_between(rad_exp, expe_avg - expe_std, expe_avg + expe_std, alpha=0.3)
        plt.xlabel("radius ($\mathrm{\mu m}$)", fontsize="large")
        plt.ylabel("$I/I_0$", fontsize="large")
        plt.legend()
        plt.title("Radial profile")

        plt.tight_layout()
        plt.show()

    def optimize(
        self,
        method="trf",
        loss="cauchy",
        report=False,
        present=False,
        radial_profile=False,
    ):
        """Fit an hologram with the given guesses"""
        # self.set_vary(self.fitter) TO DO UPDATE THIS METHOD

        self.feature.optimizer.settings["method"] = method
        self.feature.optimizer.settings["loss"] = loss
        result = self.feature.optimize()
        if report:
            self.report(result)

        if present:
            self.present(self.feature)

        if radial_profile:
            center = (result["x_p"], result["y_p"])
            self.radial_profile(self.feature.data, self.feature.hologram(), center)
        return result

    def update_guess(self, z):
        self.p = self.feature.particle.r_p[2] = z

    def fit_video(
        self,
        xc,
        yc,
        vid,
        savefile,
        h,
        n_start=1,
        n_end=None,
        method="lm",
        loss="linear",
        dark_count_mode="min",
        update_mask=True,
        percentpix = 0.1,
    ):
        """Fit a full movie by just using the guesses of the first image
        the guesses for the next image will take the precedent ones.
        Box_size is the fitted square. To initialize correctly the fitter
        it's needed to give it the the position of the crop"""
        self.xc = int(xc)
        self.yc = int(yc)
        self.h = h
        self.feature.mask.percentpix = percentpix
        if n_end == None:
            print("Computing the length of the video")
            n_end = vid.get_length()
            print("length of video = {}".format(self.number))

        image = vid.get_image(1)
        _crop_fit = self._crop_fit
        if dark_count_mode == "min":
            image = normalize(
                _crop_fit(image),
                _crop_fit(vid.background),
                dark_count=np.min(_crop_fit(image)),
            )
        elif dark_count_mode == "zero":
            image = normalize(_crop_fit(image), _crop_fit(vid.background), dark_count=0)
        elif dark_count_mode == "set":
            image = normalize(
                _crop_fit(image), _crop_fit(vid.background), dark_count=vid.dark_count
            )

        image = image / np.mean(image)
        self.fp = np.memmap(
            savefile,
            dtype="float64",
            mode="w+",
            shape=(int(n_end - n_start), len(self.feature.optimizer.variables)),
        )
        for n, i in enumerate(tqdm(range(n_start, n_end))):
            if i > n_start:
                image = normalize(vid.get_next_image(), vid.background)
                image = self._crop_fit(image)
                try:
                    image = image / np.mean(image)
                except:
                    image = image

            if update_mask:
                self.feature.mask._update()
                index = self.feature.mask.selected
                coords = self.feature.coordinates[:, index]

            self.feature.data = image
            self.result = self.optimize(method=method, loss=loss)
            self._globalize_result()
            self.save_result(n)
            self.update_guess(self.result.z_p)
        del self.fp

    def show_results(self):
        fit = self.fitter.model.hologram().reshape(self.shape)
        noise = self.fitter.noise
        mega_image = np.hstack([self.image, fit])
        fig, ax = plt.subplots(figsize=(12, 36))
        ax.imshow(mega_image, cmap="gray", interpolation=None)

    def _globalize_result(self):
        self.result.x_p = self.result.x_p + self.xc - self.h // 2
        self.result.y_p = self.result.y_p + self.yc - self.h // 2
        self.xc = int(self.result.x_p)
        self.yc = int(self.result.y_p)

    def save_result(self, n):
        buf = np.array([])
        for i in self.result[self.feature.optimizer.variables]:
            buf = np.append(buf, i)
        self.fp[n, :] = buf


def globalize_result(result, xc, yc, h):
    result.x_p = result.x_p + xc - h // 2
    result.y_p = result.y_p + yc - h // 2
    return result
