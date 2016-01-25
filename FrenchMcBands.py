import numpy as np

class FrenchMcBands(object):

    def __init__(self,i=4):
        #First digit of the run
        self.part_type=["gamma"]
        self.part_type_digit="1"
        #Second digit of the run
        self.azimuth=[180]
        self.azimuth_digit="4"
        #third digit of the run
        self.zenMC = [0, 18, 26, 32, 37, 41, 46, 50, 53, 57, 60, 63, 67, 70]
        self.zenMC_digit=["00","01","02","03","04","05","06","07","08","09","10","11","12","13"]
        #fouth digit of the run
        self.offMC = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        self.offMC_digit=["0","1","2","3","4","5"]
        #fith digit of the run
        self.energy_type="1"
        #sixth digit of the run
        self.enMC = [0.02, 0.03, 0.05, 0.08, 0.125, 0.2, 0.3, 0.5, 0.8, 1.25, 2, 3, 5, 8, 12.5, 20, 30, 50, 80, 125]
        self.enMC_digit=["01", "02" , "03" , "04", "05" ,"06" , "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        
    def find_zenMC_digit(self, zen):
        i=np.where(np.asarray(self.zenMC)==zen)[0]
        return self.zenMC_digit[i]
    
    def find_offMC_digit(self, off):
        i=np.where(np.asarray(self.offMC)==off)[0]
        return self.offMC_digit[i]
    
    def find_enMC_digit(self, E):
        i=np.where(np.asarray(self.enMC)==E)[0]
        return self.enMC_digit[i]
    
