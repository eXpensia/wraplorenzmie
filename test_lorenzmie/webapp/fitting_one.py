import sys

sys.path.append(r"C:\Users\m.lavaud\Documents\Ma_these")
import streamlit as st
import imageio
import matplotlib.pyplot as plt
import numpy as np
import wraplorenzmie.utilities.utilities as utilities
from streamlit_cropper import st_cropper
import wraplorenzmie.fits.fit as fit
from pylorenzmie.utilities import azistd
from PIL import Image

st.header("Fitting one image")


img_file = st.sidebar.file_uploader(label="Upload a file", type=["png", "jpg", "tiff"])

realtime_update = True
if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(
        img, realtime_update=realtime_update, box_color="#0000FF", aspect_ratio=(1, 1)
    )

if not img_file:
    st.stop()
    # Manipulate cropped image at will
    # st.write("Preview")
    # _ = cropped_img.thumbnail((150,150))
    # st.image(cropped_img)


def radial_profile(self, exp, theo, center):
    fig = plt.figure(figsize=(5 * 1.2, 2.5 * 1.2))
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
    st.pyplot(fig)


def _radial_profile(self, result):
    center = (result["x_p"], result["y_p"])
    radial_profile(self, self.feature.data, self.feature.hologram(), center)


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

    st.pyplot(fig)


def report(self, result):
    def value(val, err, dec=4):
        fmt = "{" + ":.{}f".format(dec) + "}"
        return (fmt + " +- " + fmt).format(val, err)

    keys = self.feature.optimizer.variables
    res = [i + " = " + value(result[i], result["d" + i]) for i in keys]
    text = ""
    text += "npixels = {}".format(result.npix) + "\n"
    for i in res:
        text += i + "\n"
    text += "chisq = {:.2f}".format(result.redchi)

    st.text(text)


def optimize(fitter):

    result = fitter.optimize()
    report(fitter, result)
    present(fitter, fitter.feature)
    _radial_profile(fitter, result)


def show_estimate(self):
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

    st.pyplot(fig)


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
    st.pyplot(fig)


im = np.array(cropped_img)

all_params = ["a_p", "n_p", "x_p", "y_p", "z_p", "n_m", "calib", "wavelength"]

free_param = st.multiselect(
    "Free parameters", all_params, default=["a_p", "n_p", "x_p", "y_p", "z_p"]
)

init_param = {}

with st.sidebar:
    st.subheader("Parameters")
    init_param["a_p"] = st.number_input("a_p", value=1.5)
    init_param["n_p"] = st.number_input("n_p", value=1.6)
    init_param["x_p"] = st.number_input("x_p", value=int(np.shape(im)[0] // 2))
    init_param["y_p"] = st.number_input("y_p", value=int(np.shape(im)[1] // 2))
    init_param["z_p"] = st.number_input("z_p", value=12.0)
    init_param["calib"] = st.number_input(
        "calib", value=0.0532, step=0.0001, format="%.4f"
    )
    init_param["wavelength"] = st.number_input(
        "wavelength", value=0.532, step=0.0001, format="%.3f"
    )
    init_param["n_m"] = st.number_input("n_m", value=1.33, step=1e-5, format="%.3f")
    percentpix = st.number_input(
        "percentpix",
        value=0.1,
        min_value=0.001,
        max_value=1.0,
        step=1e-3,
        format="%.3f",
    )


fitter = fit.fitting(
    im,
    init_param["wavelength"],
    init_param["calib"],
    n_m=init_param["n_m"],
    percentpix=percentpix,
    mask="fast",
)

fitter.set_vary(free_param)

im_fig = plt.figure(figsize=(5, 5))

selection = st.selectbox(
    "Choose what to obseve", ["show_mask", "initialize_parameters", "fit"]
)

fitter.make_guess(
    init_param["a_p"],
    init_param["n_p"],
    init_param["z_p"],
    r_p=[init_param["x_p"], init_param["y_p"]],
    show_estimate=False,
)


if selection == "show_mask":
    show_mask(fitter)
if selection == "initialize_parameters":
    show_estimate(fitter)

if selection == "fit":
    pushed = st.button("Click me to fit")
    if pushed:
        optimize(fitter)


# plt.imshow(im, cmap="gray")

# st.pyplot(im_fig)
