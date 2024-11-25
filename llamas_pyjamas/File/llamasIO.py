
from astropy.io import fits # type: ignore

##############################################################

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

##############################################################

class llamasAllCameras:

    def __init__(self, fitsfile):

        with fits.open(fitsfile) as hdulist:
            self.header     = hdulist[0].header
            self.Next       = len(hdulist) - 1
            self.extensions = []
            for hdu in hdulist[1:]:
                thiscam = llamasOneCamera()
                thiscam.readhdu(hdu)
                self.extensions.append(thiscam)

        hdulist.close()

##############################################################
            
def getBenchSideChannel(fitsfile, bench, side, channel):

    hdul = fits.open(fitsfile)
    for hdu in hdul[1:]:
        if (hdu.header['BENCH'] == bench):
            if (hdu.header['SIDE'] == side):
                if (hdu.header['COLOR']=='Color'):
                    return(hdu.data)
                


    