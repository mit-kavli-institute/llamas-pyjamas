from astropy.io import fits

class llamasOneCamera:

    def __init__(self):
        self.header =   ''
        self.data   =   0
        self.bench  =   -1
        self.side   =   ''
        self.channel =  ''

    def readhdu(self, hdu):
        self.header = hdu.header
        self.data   = hdu.data
        self.bench  = self.header['BENCH']
        self.side   = self.header['SIDE']
        self.channel = self.header['COLOR']
        self.index  = -1

    def findhdu(self, hdulist, bench=bench, side=side, channel=channel):
        for hdu in hdulist:
            if (hdu.header['BENCH'] == bench and
                hdu.header['SIDE'] == side and
                hdu.header['COLOR'] == channel):
            print ("AHA")


class llamasAllCameras:

    def __init__(self, fitsfile):

        with fits.open(fitsfile) as hdulist:
            self.header     = hdulist[0].header
            self.Next       = len(hdulist) - 1
            self.extensions = []
            for hdu in hdulist[1:]:
                if (len(self.extensions) == 0):
                    self.extensions = llamasOneCamera(hdu)
                else:
                    self.extensions.append(self.extensions, llamasOneCamera(hdu))

        hdulist.close()

            