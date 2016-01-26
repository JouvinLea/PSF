import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import iminuit
import FrenchMcBands
import PSFfit
plt.ion()

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
    khi2=np.sum(resid)/(len(x)-5)
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


#theta2min=1e-4
theta2max=0.3
#nbins=50
#theta2hist=np.logspace(np.log10(theta2min), np.log10(theta2max),nbins)

"""
MC energy, zenithal angle, offset and efficiency
"""
#enMC = [0.02, 0.03, 0.05, 0.08, 0.125, 0.2, 0.3, 0.5, 0.8, 1.25, 2, 3, 5, 8, 12.5, 20, 30, 50, 80, 125]
enMC = [0.08, 0.125, 0.2, 0.3, 0.5, 0.8, 1.25, 2, 3, 5, 8, 12.5, 20, 30, 50, 80, 125]
#enMC = [0.08]
lnenMC = np.log10(enMC)
#zenMC = [0, 18, 26, 32, 37, 41, 46, 50, 53, 57, 60, 63, 67, 70]
#effMC = [50, 60, 70, 80, 90, 100]
#offMC = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
zenMC = [18]
effMC = [70]
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


MCband=FrenchMcBands.FrenchMcBands()
directory="/Users/jouvin/Desktop/these/WorkGAMMAPI/IRF/PSF"
config="elm_south_stereo"
for (ieff, eff) in enumerate(effMC):
        print eff
        for (ioff, off) in enumerate(offMC):
            print off
            for (izen, zen) in enumerate(zenMC):
                print zen
                for (ien, E) in enumerate(enMC):
                    run_number=MCband.run_number(zen, off, E)
                    print run_number
                    PSFfile=directory+"/run_"+run_number+"_"+config+"_Prod15_4_eventlist.fits"                    
                    hdu=pf.open(PSFfile)
                    print hdu[1].header["MUONEFF"]
                    print int(hdu[4].header["OBSZEN"])
                    print hdu[4].header["HIERARCH TARGETOFFSET"]
                    print str("%.2f"%hdu[4].data["E_MIN"])
                    theta2 = hdu[1].data["MC_ThSq"]
                    Nev_perbin=10
                    Nbinmax=50
                    index = [theta2<theta2max]
                    theta2f = theta2[index]
                    Nev=len(theta2f)
                    while(Nev/Nev_perbin >  Nbinmax):
                        Nev_perbin = Nev_perbin * 2
                    theta2fsorted=np.sort(theta2f)
                    theta2hist=np.array(theta2fsorted[0])
                    index_theta2_hist=np.arange(Nev_perbin, Nev, Nev_perbin)
                    for i in index_theta2_hist:
                        #print theta2hist_variablesize
                        theta2hist = np.append(theta2hist, theta2fsorted[i])                        
                    #MINIMIZATION
                    hist, bin_edges = np.histogram(theta2,theta2hist)
                    PSF=PSFfit.PSFfit(theta2f)
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
                    TableSigma1[ien, ioff, izen, ieff] = s1
                    TableSigma2[ien, ioff, izen, ieff] = s2
                    TableSigma3[ien, ioff, izen, ieff] = s3
                    TableA2[ien, ioff, izen, ieff] = A2
                    TableA3[ien, ioff, izen, ieff] = A3
                    #If the energy bin are in log, we have to take sqrt(Emin*Emax) for the center of the bin
                    #theta2bin = np.sqrt(bin_edges[:-1] * bin_edges[1:])
                    theta2bin = (bin_edges[:-1] + bin_edges[1:])/2.
                    #thetafit=np.logspace(np.log10(theta2bin[0]),np.log10(theta2bin[-1]),100)
                    #fitfun = triplegauss(thetafit,s1,s2,s3,A2,A3)
                    #We have to divide by the solid angle of each bin= pi*dO^2 to normalize the histogram
                    bsize = np.diff(bin_edges)*math.pi
                    hist_norm = hist/float(np.sum(hist))/bsize
                    # use gehrels errors for low counts (http://cxc.harvard.edu/sherpa4.4/statistics/)
                    hist_err = (1+np.sqrt(hist+0.75))/float(np.sum(hist))/bsize
                    i_nonnulle=np.where(hist_norm!=0)
                    fitfun = lambda x : triplegauss(x,s1,s2,s3,A2,A3)
                    save_fig="fitspsf_run_"+run_number+".jpg"
                    plot_fit_delchi(theta2bin[i_nonnulle],hist_norm[i_nonnulle],hist_err[i_nonnulle],fitfun,save_fig)
                    print "khi2= ", khi2(theta2bin,hist_norm,hist_err,fitfun)
                    #i_sup,List_sup,i_inf,List_inf=bin_contigu(theta2bin,hist_norm,fitfun,1/25.)
                    np.savez("PSF_"+config+".npz", TableSigma1=TableSigma1, TableSigma2=TableSigma2, TableSigma3=TableSigma3, TableA2=TableA2, TableA3=TableA3)
                        
