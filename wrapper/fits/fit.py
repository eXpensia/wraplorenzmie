import pylorenzmie

class fitter(object):
    ''' All the wrap aroud the pylorenzmie toolbox
    available at the grier lab github
    https://github.com/davidgrier/pylorenzmie '''

    def __init__(image,wavelength,magnification,n_m=1.33,
                    dark_count=0,background=1):
        self.image = image
        self.shape = image.shape
        self.instrument = Instrument.Instrument(wavelength=wavelength
                                    magnification=magnification, #microns/pixel
                                    n_m=n_m,
                                    dark_count=dark_count,
                                    background=background)


    def make_guess(self,a_p,n_p,z):
        self.guesses =
        guesses = {'a_p': .92, #um
                   'n_p': 1.41,
                   'r_p': [shape[0] // 2, shape[1] // 2, z]} #pixels
