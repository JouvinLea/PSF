import numpy as np
import pyfits
from glob import glob
import math
import pylab as pt
import ROOT
import root_numpy as rn
from root_numpy import array2hist, fill_hist
from scipy.optimize import curve_fit
import iminuit
pt.ion()

#Pour l intant que pour ce fichier car la fonction nllp_triplegauss a besoin des thetaobs comme variable globale donc va falloir voir comment on peut faire pour une boucle sur les observations
file="run_14012110_elm_south_stereo_Prod15_4_eventlist.fits"
hdu=pyfits.open(file)
theta2obs=hdu[1].data["MC_ThSq"]
i=np.where(theta2obs<0.1)
theta2obs_good=theta2obs[i]

theta2min=0.
theta2max=0.3
theta2=np.linspace(theta2min, theta2max,200)
    
def triple_gauss(thetasquare,S,s1,A2,s2,A3,s3):
    y=(S/math.pi)*(np.exp(-thetasquare/(2*s1**2))+A2*np.exp(-thetasquare/(2*s2**2))+A3*np.exp(-thetasquare/(2*s3**2)))
    return y

def triple_gauss_norm(thetasquare,s1,A2,s2,A3,s3):
    norm=2*math.pi*(s1**2+A2*s2**2+A3*s3**2)
    y=(1/norm)*(np.exp(-thetasquare/(2*s1**2))+A2*np.exp(-thetasquare/(2*s2**2))+A3*np.exp(-thetasquare/(2*s3**2)))
    return y

def nllp_triplegauss(s1,A2,s2,A3,s3):
    norm=2*math.pi*(s1**2+A2*s2**2+A3*s3**2)
    y=np.sum(np.log(1/norm*(np.exp(-theta2obs_good/(2*s1**2))+A2*np.exp(-theta2obs_good/(2*s2**2))+A3*np.exp(-theta2obs_good/(2*s3**2)))))
    return y

             
m=iminuit.Minuit(nllp_triplegauss, s1=0.02, A2=4, s2=0.05, A3=4,s3=0.08)
#m=iminuit.Minuit(nllp_triplegauss, s1=0.02, A2=4, s2=0.05, A3=4,s3=0.08,limit_s1=(-0.1,0.1),limit_s2=(-0.1,0.1),limit_s3=(-0.1,0.1))
#m=iminuit.Minuit(nllp_triplegauss, s1=0.02, A2=4, s2=0.05, A3=4,s3=0.08)
#m.migrad()

#Definition de la triple gaussienne pour root
#x=thetasquare
myfit_triple_gauss=ROOT.TF1("TripleGauss","([0]/TMath::Pi())*(exp(-x/(2*[1]*[1]))+[2]*exp(-x/(2*[3]*[3]))+[4]*exp(-x/(2*[5]*[5])))", 0, 2);

#ListRun = glob('run*.fits')
ListRun = ["run_14012110_elm_south_stereo_Prod15_4_eventlist.fits"]
for file in ListRun:
    hdu=pyfits.open(file)
    print file
    print hdu[1].header["MUONEFF"]
    print int(hdu[4].header["OBSZEN"])
    print hdu[4].header["HIERARCH TARGETOFFSET"]
    print str("%.2f"%hdu[4].data["E_MIN"])
    theta2obs=hdu[1].data["MC_ThSq"]
    hist, bin_edges = np.histogram(theta2obs,theta2)
    #Le x il faut creer le milieu du bin je pense!
    #popt, pcov = curve_fit(triple_gauss, theta2[0:-1],hist)
    popt, pcov = curve_fit(triple_gauss, theta2[0:-1],hist,[2500,0.02,4,0.05,4,0.08])
    fit=triple_gauss(theta2[0:-1],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
    pt.figure(1)
    pt.plot(theta2[0:-1], hist, label="hist")
    pt.plot(theta2[0:-1], fit, label= "fit")
    pt.xlabel("theta2")
    pt.yscale("log")
    pt.xscale("log")
    pt.legend()

    """
    histroot= ROOT.TH1F("Root_psf_histo","Root_psf_histo", 199,theta2min ,theta2max)
    rn.fill_hist(histroot,theta2obs)
    myfit_triple_gauss.SetParameter(0,2500)
    myfit_triple_gauss.SetParameter(1,0.02)
    myfit_triple_gauss.SetParameter(2,4)
    myfit_triple_gauss.SetParameter(3,0.05)
    myfit_triple_gauss.SetParameter(4,4)
    myfit_triple_gauss.SetParameter(5,0.08)
    histroot.Fit("TripleGauss", "L")
    fitroot=histroot.GetFunction("TripleGauss");
    print "khi2= " ,fitroot.GetChisquare()
    fitroot.GetParameter(0)
    fitroot.GetParameter(1)
    fitroot.GetParameter(2)
    fitroot.GetParameter(3)
    fitroot.GetParameter(4)
    fitroot.GetParameter(5)
    #fitroot.GetParError(1)
    """
    #m=iminuit.Minuit(nllp_triplegauss)
    #m.migrad()

s1=m.values['s1']
s2=m.values['s2']
s3=m.values['s3']
A2=m.values['A2']
A3=m.values['A3']
norm=2*math.pi*(s1**2+A2*s2**2+A3*s3**2)
bb=triple_gauss_norm(theta2obs_good,s1,A2,s2,A3,s3)*norm
pt.plot(theta2obs_good,bb, "+")
pt.plot(theta2[0:-1], hist/float(np.sum(hist)), label="hist")
pt.legend()
"""
S=2500
s1=0.02
A2=4
s2=0.05
A3=4
s3=0.08
y=nllp_triplegauss(S,s1,A2,s2,A3,s3)
#pt.plot(theta2,y)
"""
