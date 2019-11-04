import imageio
import scipy.ndimage.gaussian_filter as gaussain_filter

class video_reader(object):

    def __init__(self, filename, codecs = "ffmpeg"):
        self.filename = filename
        self.codecs = codecs
        self.vid = imageio.get_reader(self.filename,  self.codecs)
        self.length =  self.vid.get_length()

    def close(self):
        self.vid.close()

    def get_image(self,n):
        ''' Get the image n of the movie '''
        return self.vid.get_data(n)[:,:,1]

    def get_filtered_image(self,n,sigma):
        ''' Get the image n of the movie and apply a gaussian filter '''
        return gaussian_filter(self.vid.get_data(n)[:,:,1], sigma = sigma)

    def get_background(self,n):
        ''' Compute the background over n images '''
        image = get_image(1)
        size = (get_image.shape[0],get_image.shape[1],n)
        buf = np.empty(size)
        for i,toc in enumerate(np.arrange(0,self.length, self.length // (n-1))):
            buf[:,:,i] = get_image(toc)

        return np.mean(buf, axis=0)
