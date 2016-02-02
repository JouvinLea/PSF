import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import FrenchMcBands
import PSFfit
from scipy.special import erf
from astropy.stats import poisson_conf_interval
plt.ion()
from scipy import stats

"""
Fonction defenition
"""

def triplegauss(theta2,s1,s2,s3,A2,A3):
    s12 = s1*s1
    s22 = s2*s2
    s32 = s3*s3
    
    gaus1 = np.exp(-theta2/(2*s12))
    gaus2 = np.exp(-theta2/(2*s22))
    gaus3 = np.exp(-theta2/(2*s32))

    y = (gaus1 + A2*gaus2 + A3*gaus3) 
    norm =  2*math.pi*(s12+ np.abs(A2) * s22 + np.abs(A3) * s32)
    return y/norm

def Integral_triplegauss(theta1,theta2,s1,s2,s3,A2,A3):
    s12 = s1*s1
    s22 = s2*s2
    s32 = s3*s3
    
    gaus1 = erf(theta2/(np.sqrt(2)*s1))-erf(theta1/(np.sqrt(2)*s1))
    gaus2 = erf(theta2/(np.sqrt(2)*s2))-erf(theta1/(np.sqrt(2)*s2))
    gaus3 = erf(theta2/(np.sqrt(2)*s3))-erf(theta1/(np.sqrt(2)*s3))

    y = np.sqrt(math.pi/2)*(gaus1 + A2*s2*gaus2 + A3*s3*gaus3) 
    norm =  2*math.pi*(s12+ np.abs(A2) * s22 + np.abs(A3) * s32)
    return y/norm

def king(theta2,sig, gam):
    norm = (1/(2*np.pi*sig**2))*(1-1/gam)
    king=(1+theta2/(2*gam*(sig**2)))**(-gam)
    return norm*king

def bin_contigu(x,data,model_fun, threshold):
    i_sup=np.where(data > model_fun(x))[0]
    i_inf=np.where(data < model_fun(x))[0]
    List_sup=[]
    List_inf=[]
    iband=0
    for i,i_s in enumerate(i_sup.tolist()):
        if(i==0):
            List_sup.append([i_s])
        elif(i_s == i_sup[i-1]+1):
            List_sup[iband].append(i_s)
        else:
            List_sup.append([i_s])
            iband += 1
    iband=0
    for i,i_i in enumerate(i_inf.tolist()):
        if(i==0):
            List_inf.append([i_i])
        elif(i_i == i_inf[i-1]+1):
            List_inf[iband].append(i_i)
        else:
            List_inf.append([i_i])
            iband += 1
    Npoints=len(x)
    Nsup=len(List_sup)
    Ninf=len(List_inf)
    for i in range(Nsup):
        if(len(List_sup[i])>= threshold*Npoints):
            print "WARNING: There are ",len(List_sup[i]) ," values that are superior to the fit"
    for i in range(Ninf):
        if(len(List_inf[i])>= threshold*Npoints):
            print "WARNING: There are ",len(List_inf[i]) ," values that are inferiror to the fit"
        
    return i_sup,List_sup,i_inf,List_inf

# Figure definitions
def khi2(x,data,err,model_fun):
    resid = (data - model_fun(x))**2/err**2
    khi2=np.sum(resid/len(x))
    return khi2

def khi2_int(x,data,err,model_fun):
    resid = (data - model_fun)**2/err**2
    khi2=np.sum(resid/len(x))
    return khi2
       
def plot_fit_delchi(x,data,err,model_fun,save_fig):
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:3,:]) # rows, cols, plot_num.
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ax1.errorbar(x,data,yerr=err,fmt='o',color='k')
    xmod = np.linspace(np.min(x),np.max(x),10000)
    KHI2=khi2(x, data, err, model_fun)
    line1,=ax1.plot(x,model_fun(x))
    ax1.plot(x,model_fun(x))
    ax1.get_xaxis().set_visible(False)
    plt.legend([line1], ["khi2= "+str("%.2f"%KHI2)])

    ax2 = fig.add_subplot(gs[3,:],sharex=ax1) 
    ax2.plot(xmod,np.zeros_like(xmod),color='k')
    resid = (data - model_fun(x))/err
    ax2.errorbar(x, resid, yerr=np.ones_like(x), fmt='o', color='k')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(save_fig)

def plot_fit_delchi_int(x,data,err,model_fun,save_fig):
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:3,:]) # rows, cols, plot_num.
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ax1.errorbar(x,data,yerr=err,fmt='o',color='k')
    xmod = np.linspace(np.min(x),np.max(x),10000)
    KHI2=khi2_int(x, data, err, model_fun)
    line1,=ax1.plot(x,model_fun)
    ax1.plot(x,model_fun)
    ax1.get_xaxis().set_visible(False)
    plt.legend([line1], ["khi2= "+str("%.2f"%KHI2)])

    ax2 = fig.add_subplot(gs[3,:],sharex=ax1) 
    ax2.plot(xmod,np.zeros_like(xmod),color='k')
    resid = (data - model_fun)/err
    ax2.errorbar(x, resid, yerr=np.ones_like(x), fmt='o', color='k')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(save_fig)
    
def plot_fit_delchi_test(x,data,err,model_fun,save_fig):
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:3,:]) # rows, cols, plot_num.
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ax1.errorbar(x,data,yerr=err,fmt='o',color='k')
    xmod = np.linspace(np.min(x),np.max(x),10000)
    KHI2=khi2(x, data, err, model_fun, N_param_model)
    line1,=ax1.plot(x,model_fun(x))
    ax1.plot(x,model_fun(x))
    ax1.get_xaxis().set_visible(False)
    plt.legend([line1], ["khi2= "+str("%.2f"%KHI2)])

    #ax2 = fig.add_subplot(gs[3,:],sharex=ax1) 
    #ax2.plot(xmod,np.zeros_like(xmod),color='k')
    #resid = (data - model_fun(x))/err
    #ax2.errorbar(x, resid, yerr=np.ones_like(x), fmt='o', color='k')
    #plt.subplots_adjust(hspace=0.1)
    plt.savefig(save_fig)
    
def theta2_bin(data, Nev_bin, Nbinmax):
    """
    Define an adaptative theta2binning in order to have at least 10 evnts per bin
    """
    Nev=len(data)
    while(Nev/Nev_bin >  Nbinmax):
        Nev_bin = Nev_bin * 2
    theta2sorted=np.sort(data)
    theta2bin=np.array(theta2sorted[0])
    index=np.arange(Nev_bin, Nev, Nev_bin)
    for i in index:
        theta2bin = np.append(theta2bin, theta2sorted[i]) 
    return theta2bin



#theta2min=1e-4
theta2max=0.3
#nbins=50
#theta2hist=np.logspace(np.log10(theta2min), np.log10(theta2max),nbins)

"""
MC energy, zenithal angle, offset and efficiency
"""
#enMC = [0.02, 0.03, 0.05, 0.08, 0.125, 0.2, 0.3, 0.5, 0.8, 1.25, 2, 3, 5, 8, 12.5, 20, 30, 50, 80, 125]
enMC = [0.08, 0.125, 0.2, 0.3, 0.5, 0.8, 1.25, 2, 3, 5, 8, 12.5, 20, 30, 50, 80, 125]
#enMC = [0.08, 0.5, 0.8,0.125, 1.25, 80, 125]
#enMC = [0.125]
#lnenMC = np.log10(enMC)
#zenMC = [0, 18, 26, 32, 37, 41, 46, 50, 53, 57, 60, 63, 67, 70]
#effMC = [50, 60, 70, 80, 90, 100]
#offMC = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
zenMC = [0, 18]
#zenMC = [0]
effMC = [100]
offMC = [1.0]

binEMC = len(enMC)
binzen = len(zenMC)
binoff = len(offMC)
bineff = len(effMC)


# Size of the 4D table where we will stock parameters of the tripple gaussian fit on the MC data for each true energy, zenithal angle, offset and efficiency of the MCs
TableSigma1 = np.zeros((binEMC, binoff, binzen, bineff))
TableSigma2 = np.zeros((binEMC, binoff, binzen, bineff))
TableSigma3 = np.zeros((binEMC, binoff, binzen, bineff))
TableA2 = np.zeros((binEMC, binoff, binzen, bineff))
TableA3 = np.zeros((binEMC, binoff, binzen, bineff))
TableSig = np.zeros((binEMC, binoff, binzen, bineff))
TableGam = np.zeros((binEMC, binoff, binzen, bineff))

MCband=FrenchMcBands.FrenchMcBands()
directory="/Users/jouvin/Desktop/these/WorkGAMMAPI/IRF/PSF"
config="elm_south_stereo"
file_nosimu = open("file_nosimu.txt", "w")
file_toofewevents = open("file_toofewevents.txt", "w")
file_nosimu.write("Runnumber \t E(Tev) \t Zen \t theta \t Eff \n")
file_toofewevents.write("Runnumber \t E(Tev) \t Zen \t theta \t Eff \t Number of events \n") 
for (ieff, eff) in enumerate(effMC):
        print eff
        for (ioff, off) in enumerate(offMC):
            print off
            for (izen, zen) in enumerate(zenMC):
                print zen
                s1_init=0.02
                s2_init=0.05
                s3_init=0.08
                A2_init=0.3
                A3_init=0.1
                for (ien, E) in enumerate(enMC):
                    run_number=MCband.run_number(zen, off, E)
                    print run_number
                    #PSFfile=directory+"/run_"+run_number+"_"+config+"_Prod15_4_eventlist.fits"
                    PSFfile=directory+"/run_"+run_number+"_Eff"+str(eff)+"_psf.fits"
                    try: 
                        hdu=pf.open(PSFfile)
                        
                    except:
                        print("Cannot open file: " + PSFfile)
                        print("skipping run")
                        E=MCband.ener_MC(run_number)
                        zen=MCband.zen_MC(run_number)
                        off=MCband.off_MC(run_number)
                        file_nosimu.write(run_number+"\t"+str(E)+"\t"+str(zen)+"\t"+str(off)+"\t"+str(eff)+"\n") 
                        continue
                    
                    #print hdu[1].header["MUONEFF"]
                    #print int(hdu[4].header["OBSZEN"])
                    #print hdu[4].header["HIERARCH TARGETOFFSET"]
                    #print str("%.2f"%hdu[4].data["E_MIN"])
                    theta2 = hdu[1].data["MC_ThSq"]
                    index = [theta2<theta2max]
                    theta2f = theta2[index]
                    if(len(theta2f)<20):
                        E=MCband.ener_MC(run_number)
                        zen=MCband.zen_MC(run_number)
                        off=MCband.off_MC(run_number)
                        file_toofewevents.write(run_number+"\t"+str(E)+"\t"+str(zen)+"\t"+str(off)+"\t"+str(eff)+"\t"+str(len(theta2f))+"\n") 
                        continue
                    Nev_perbin=10
                    Nbinmax=50
                    theta2hist=theta2_bin(theta2f, Nev_perbin, Nbinmax)
                    hist, bin_edges = np.histogram(theta2,theta2hist)
                    #Me renvois la valeur moyenne en theta2 des evenements stockes dans les bins donc peut etre un peu mieux que de prendre thetabi=(Emin+Emax)/2
                    theta2bintest, bin_edgestest,a = stats.binned_statistic(theta2,theta2,'mean',theta2hist)
                    histtest, bin_edgestest,b = stats.binned_statistic(theta2,theta2,'count',theta2hist)
                    PSF=PSFfit.PSFfit(theta2f)
                    s1,s2,s3,A2,A3=PSF.minimization("triplegauss",s1_init, s2_init, s3_init, A2_init, A3_init)
                    s1_init=s1
                    s2_init=s2
                    s3_init=s3
                    A2_init=A2
                    A3_init=A3
                    #sig,gam=PSF.minimization("king")
                    TableSigma1[ien, ioff, izen, ieff] = s1
                    TableSigma2[ien, ioff, izen, ieff] = s2
                    TableSigma3[ien, ioff, izen, ieff] = s3
                    TableA2[ien, ioff, izen, ieff] = A2
                    TableA3[ien, ioff, izen, ieff] = A3
                    #TableSig[ien, ioff, izen, ieff] = sig
                    #TableGam[ien, ioff, izen, ieff] = gam
                    np.savez("PSF_triplegauss_"+config+".npz", TableSigma1=TableSigma1, TableSigma2=TableSigma2, TableSigma3=TableSigma3, TableA2=TableA2, TableA3=TableA3)
                    #np.savez("PSF_king_"+config+".npz", TableSig=TableSig, TableGam=TableGam)  
                    #If the energy bin are in log, we have to take sqrt(Emin*Emax) for the center of the bin
                    #theta2bin = np.sqrt(bin_edges[:-1] * bin_edges[1:])
                    theta2bin = (bin_edges[:-1] + bin_edges[1:])/2.
                    #thetafit=np.logspace(np.log10(theta2bin[0]),np.log10(theta2bin[-1]),100)
                    #We have to divide by the solid angle of each bin= pi*dO^2 to normalize the histogram
                    bsize = np.diff(bin_edges)*math.pi
                    hist_norm = hist/float(np.sum(hist))/bsize
                    bsizetest = np.diff(bin_edgestest)*math.pi
                    hist_normtest = histtest/float(np.sum(histtest))/bsizetest
                    # use gehrels errors for low counts (http://cxc.harvard.edu/sherpa4.4/statistics/)
                    hist_err = (1+np.sqrt(hist+0.75))/float(np.sum(hist))/bsize
                    hist_err2 = (1+np.sqrt(histtest+0.75))/float(np.sum(histtest))/bsizetest
                    #Erreur prenant en compte poisson du coup j ai des erreurs asymetrics inf et sup
                    histerr_test=poisson_conf_interval(hist)/float(np.sum(hist))/bsize
                    fitgauss = lambda x : triplegauss(x,s1,s2,s3,A2,A3)
                    Int_fitgauss = lambda x1,x2 : Integral_triplegauss(x1,x2,s1,s2,s3,A2,A3)
                    
                    #fitking = lambda x : king(x,sig,gam)
                    save_fig="plot2/triplegauss_fitspsf_run_"+run_number+"_eff_"+str(eff)+".jpg"
                    plot_fit_delchi(theta2bin,hist_norm,hist_err,fitgauss,save_fig)
                    save_fig_int="plot2/INT_triplegauss_fitspsf_run_"+run_number+"_eff_"+str(eff)+".jpg"
                    thetamin=np.sqrt(bin_edges[:-1])
                    thetamax=np.sqrt(bin_edges[1:])
                    #plot_fit_delchi_int(theta2bin,hist_norm,hist_err,Int_fitgauss (thetamin,thetamax) ,save_fig)
                    #save_fig_scipy="plot/triplegauss_fitspsf_run_"+run_number+".jpg"
                    #plot_fit_delchi(theta2bintest,hist_normtest,hist_err2,fitgauss,save_fig_scipy,5)
                    #save_fig3="plot/triplegauss_poissonianerror_fitspsf_run_"+run_number+".jpg"
                    #plot_fit_delchi_test(theta2bin,hist_norm,histerr_test,fitgauss,save_fig3,5)
                    #save_fig2="plot/king_fitspsf_run_"+run_number+".jpg"
                    #plot_fit_delchi(theta2bin,hist_norm,hist_err,fitking,save_fig2, 2)
                    test_gauss=bin_contigu(theta2bin,hist_norm,fitgauss, 1/3.)
                    #test_king=bin_contigu(theta2bin,hist_norm,fitking, 1/3.)
file_toofewevents.close()
file_nosimu.close()
