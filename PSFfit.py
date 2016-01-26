import numpy as np

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
