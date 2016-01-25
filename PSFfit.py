import numpy as np
import pyfits
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import iminuit
import pylab as pt
import math
from glob import glob
pt.ion()

def triplegauss(theta2,s1,s2,s3,A2,A3):
    s12 = s1*s1
    s22 = s2*s2
    s32 = s3*s3
    
    gaus1 = np.exp(-theta2/(2*s12))
    gaus2 = np.exp(-theta2/(2*s22))
    gaus3 = np.exp(-theta2/(2*s32))

    y = (gaus1 + A2*gaus2 + A3*gaus3) 
    norm =  2*(s12+ np.abs(A2) * s22 + np.abs(A3) * s32)
    return y/norm
    #return y

class PSFfit(object):
    
    def __init__(self, theta2):
        self.theta2=theta2

    def nllp_triplegauss(self,s1,s2,s3,A2,A3):
        s12 = np.min([s1,s2,s3])**2
        s22 = np.median([s1,s2,s3])**2
        s32 = np.max([s1,s2,s3])**2
        norm = 2*np.pi*(s12+ np.abs(A2) * s22 + np.abs(A3) * s32)
        gaus1 = np.exp(-self.theta2/(2*s12))
        gaus2 = np.exp(-self.theta2/(2*s22))
        gaus3 = np.exp(-self.theta2/(2*s32))

        lnlike = np.log(gaus1 + A2*gaus2 + A3*gaus3) - np.log(norm)
        
        res = - lnlike.sum()

        return res

theta2min=1e-4
theta2max=0.3
nbins=50
theta2hist=np.logspace(np.log10(theta2min), np.log10(theta2max),nbins)
#theta2hist=np.linspace(theta2min, theta2max,nbins)

#ListRun = glob('run*.fits')
ListRun = ["run_14012110_elm_south_stereo_Prod15_4_eventlist.fits"]
#ListRun = ["run_14012105_elm_south_stereo_Prod15_4_eventlist.fits"]
for (i,file) in enumerate(ListRun):
    hdu=pyfits.open(file)
    hdu=pf.open(file)
    theta2 = hdu[1].data["MC_ThSq"]
    index = [theta2<theta2max]
    theta2f = theta2[index]

    hist, bin_edges = np.histogram(theta2,theta2hist)
    PSF=PSFfit(theta2f)
    #m=iminuit.Minuit(PSF.nllp_triplegauss, s1=0.02, s2=0.05,s3=0.08, A2=0.3,A3=0.1, limit_A2 = (1e-10,10.),limit_A3 = (1e-10,10.),limit_s1 = (0.005,0.1), limit_s2 = (0.005,0.2),limit_s3 = (0.02,0.5))
    m=iminuit.Minuit(PSF.nllp_triplegauss, s1=0.02, s2=0.05,s3=0.08, A2=0.3,A3=0.1,
                 limit_A2 = (1e-10,10.),limit_A3 = (1e-10,10.),
                 limit_s1 = (0.005,0.1), limit_s2 = (0.005,0.2),
                 limit_s3 = (0.02,0.5))
    m.migrad()

    s1_m=m.values['s1']
    s2_m=m.values['s2']
    s3_m=m.values['s3']

    s1 = np.min([s1_m,s2_m,s3_m])
    s2 = np.median([s1_m,s2_m,s3_m])
    s3 = np.max([s1_m,s2_m,s3_m])
    A2=m.values['A2']
    A3=m.values['A3']

    #theta2bin = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #thetafit=np.linspace(theta2bin[0],theta2bin[-1],100)
    theta2bin = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    thetafit=np.logspace(np.log10(theta2bin[0]),np.log10(theta2bin[-1]),100)
    fitfun = triplegauss(thetafit,s1,s2,s3,A2,A3)
    #fitfun = triplegauss(theta2bin,s1,s2,s3,A2,A3)
    #hist_norm = hist/float(np.sum(hist))*2
    #hist_err = np.sqrt(hist)/float(np.sum(hist))*2
    #hist_norm = hist/float(np.sum(hist))
    #hist_err = np.sqrt(hist)/float(np.sum(hist))
    bsize = np.diff(bin_edges)
    hist_norm = hist/float(np.sum(hist))/bsize
    # use gehrels errors for low counts
    hist_err = (1+np.sqrt(hist+0.75))/float(np.sum(hist))/bsize
    print file
    print i
    pt.figure(i)
    pt.loglog(thetafit, fitfun , label="fit")
    #pt.loglog(theta2bin, fitfun , label="fit")
    #pt.loglog(theta2bin, hist_norm*2, label="histnorm")
    pt.errorbar(theta2bin, hist_norm,yerr = hist_err, label="hist")
    #pt.plot(theta2hist[0:-1], hist, label="hist")
    pt.legend()
    pt.savefig("fitspsf_run_"+file[4:12]+".jpg")

