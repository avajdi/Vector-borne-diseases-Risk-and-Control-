# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:35:19 2025

@author: avajdi
"""

import matplotlib.pyplot as plt
import numpy as np

def CUL_PIP_MOL_egg_dev_mor(temp):
    Sr=np.zeros(temp.shape)    
    dum=(temp>=1) & (temp<=37)
    Qp=np.array([0.2055,   -0.0251,   37.4613])
    Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
    Sr[~dum]=1
    Sr0=1
    
    Dl=np.zeros(temp.shape)
    dum=temp<=36
    Lp=np.array([0.1695, 6.2512])
    Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
    Dl[~dum]=-Lp[0] * 36+Lp[1]
    Dl0=-Lp[0] *1+Lp[1]
    
    gamd=np.zeros(temp.shape)
    dum=temp>=1
    gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
    gamd[~dum]=-np.log(Sr0/100)/Dl0
    
    return Dl, Sr, gamd
  

    
    
def CUL_PIP_MOL_L1_dev_mor(temp):
    Sr=np.zeros(temp.shape)    
    dum=(temp>=1) & (temp<=33)
    Bp=np.array([  0.0483,   -0.8899,   33.2247])
    Sr[dum] = Bp[0] * temp[dum] * (temp[dum] - Bp[1]) * np.sqrt(Bp[2]- temp[dum])
    Sr[~dum]=0.25
    Sr0=0.25
    
    Dl=np.zeros(temp.shape)
    dum=temp<=34
    Lp=np.array([0.6700 , 22.9750])
    Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
    Dl[~dum]=-Lp[0] * 34+Lp[1]
    Dl0=-Lp[0] *1+Lp[1]
    
    gamd=np.zeros(temp.shape)
    dum=temp>=1
    gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
    gamd[~dum]=-np.log(Sr0/100)/Dl0
    
    return Dl, Sr, gamd 
   
    

    
def CUL_PIP_MOL_L2_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=-2) & (temp<=42)
        Qp=np.array([ 0.1974,   -2.4103 ,  42.1515])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1  
        
        Dl=np.zeros(temp.shape)
        dum=temp<=36
        Lp=np.array([0.3200,   11.7500])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 36+Lp[1]
        Dl0=-Lp[0] * (-2) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=-2
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 

   
    
def CUL_PIP_MOL_L3_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=-2) & (temp<=43)
        Qp=np.array([0.1868 ,  -2.5918,   43.3909])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1
        
        Dl=np.zeros(temp.shape)
        dum=temp<=34
        Lp=np.array([ 0.3457,   11.9607])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 34+Lp[1]       
        Dl0=-Lp[0] * (-2) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=-2
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        
    
    
def CUL_PIP_MOL_L4_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=-1) & (temp<=42)
        Qp=np.array([ 0.2000,   -1.8528,   42.1471])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1
        
        Dl=np.zeros(temp.shape)
        dum=temp<=36
        Lp=np.array([   0.4100 ,  15.0750])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 36+Lp[1]        
        Dl0=-Lp[0] * (-1) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=-1
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
    
    
def CUL_PIP_MOL_Pup_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=1) & (temp<=32)
        Bp=np.array([0.0103, -105.5686,   32.8422])
        Sr[dum] = Bp[0] * temp[dum] * (temp[dum] - Bp[1]) * np.sqrt(Bp[2]- temp[dum])
        Sr[~dum]=1
        Sr0=1
         
        Dl=np.zeros(temp.shape)
        dum=temp<=34
        Lp=np.array([0.2743,   9.4643])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 34+Lp[1]      
        Dl0=-Lp[0] * (1) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=1
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        
    
    
def CUL_PIP_MOL_Adl_longe(temp):
        Dl=np.zeros(temp.shape)   
        dum1=(temp>=15) & (temp<=32)
        dum2=(temp>=10) & (temp<15)
        dum3=(temp>=0) & (temp<10)
        dum4=(temp<0) | (temp>32)

        Lp=np.array([3.9943 , 129.5123])
        Dl[dum1]=-Lp[0] * temp[dum1]+Lp[1]
        Dl[dum2]=-Lp[0] * 15+Lp[1]
        Dl[dum3]=(-Lp[0] * 15+Lp[1]-1)*temp[dum3]/10+1
        Dl[dum4]=1
        
        return Dl  
    
    
def CUL_PIP_PIP_gonc(temp):
        Dl_inv=np.zeros(temp.shape)    
        dum=(temp>=8.5) & (temp<=39.6)
        Bp=np.array([  1.7e-4,   8.5,   39.6])
        Dl_inv[dum] = Bp[0] * temp[dum] * (temp[dum] - Bp[1]) * np.sqrt(Bp[2]- temp[dum])
        Dl_inv[~dum]=0
        
        return Dl_inv


def CUL_PIP_PIP_ov(temp):
        ov=np.zeros(temp.shape)    
        dum=(temp>=5.2) & (temp<=33.5)
        Qp=np.array([0.4755,    5.0606,   33.8633])
        ov[dum] = Qp[0] * (temp[dum] - Qp[1]) * (Qp[2]- temp[dum])
        ov[~dum]=0
        
        return ov

    
def CUL_PIP_PIP_Adl_longe(temp):
        Dl=np.zeros(temp.shape)   
        dum1=(temp>=8) & (temp<=32)
        dum2=(temp>=3) & (temp<8)
        dum3=(temp>=0) & (temp<3)
        dum4=(temp<0) | (temp>32)

        Lp=np.array([3.9943 , 129.5123])
        Dl[dum1]=-Lp[0] * temp[dum1]+Lp[1]
        Dl[dum2]=-Lp[0] * 8+Lp[1]
        Dl[dum3]=(-Lp[0] * 8+Lp[1]-1)*temp[dum3]/3+1
        Dl[dum4]=1
        
        return Dl  
    

def CUL_PIP_PIP_Adl_longe_di(temp):
        Dl=np.zeros(temp.shape)   
        dum1=(temp>=7) & (temp<=32)
        dum2= (temp<7)
        dum3=(temp>32)

        Lp=np.array([3.9943 , 129.5123])
        Dl[dum1]=-Lp[0] * temp[dum1]+Lp[1]
        Dl[dum2]=-Lp[0] * 7+Lp[1]
        Dl[dum3]=1
        
        return Dl  


def CUL_PIP_PIP_L1_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=0) & (temp<=38)
        Qp=np.array([ 0.24 ,  -1.0790 ,  39.4297])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1
        
        Dl=np.zeros(temp.shape)
        dum=temp<=31
        Lp=np.array([ 0.1888 ,   6.9898])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 31+Lp[1]        
        Dl0=-Lp[0] * (0) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=0
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        


def CUL_PIP_PIP_L2_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=0) & (temp<=39)
        Qp=np.array([ 0.2421,   -0.8308,   39.4748])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1
        
        Dl=np.zeros(temp.shape)
        dum=temp<=28
        Lp=np.array([ 0.3940 ,  12.1810])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 28 + Lp[1]       
        Dl0=-Lp[0] * (0) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=0
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        


def CUL_PIP_PIP_L3_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=-1) & (temp<=37)
        Qp=np.array([ 0.2298 ,  -1.8152 ,  38.1061])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1   
        
        Dl=np.zeros(temp.shape)
        dum=temp<=30
        Lp=np.array([0.2855 ,   9.6680])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 30 + Lp[1]       
        Dl0=-Lp[0] * (-1) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=-1
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        



def CUL_PIP_PIP_L4_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=-1) & (temp<=35)
        Qp=np.array([ 0.2779,   -1.0873 ,  35.2593])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1  
        
        Dl=np.zeros(temp.shape)
        dum=temp<=30
        Lp=np.array([0.7082 ,  22.9061])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 30 + Lp[1]        
        Dl0=-Lp[0] * (-1) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=-1
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        

    
    
def CUL_PIP_PIP_Pup_dev_mor(temp):
        Sr=np.zeros(temp.shape)    
        dum=(temp>=4) & (temp<=37)
        Qp=np.array([ 0.2849 ,   3.8794 ,  37.2919])
        Sr[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        Sr[~dum]=1
        Sr0=1  
        
        Dl=np.zeros(temp.shape)
        dum=temp<=33
        Lp=np.array([0.2686 ,   9.2200])
        Dl[dum]=-Lp[0] * temp[dum]+Lp[1]
        Dl[~dum]=-Lp[0] * 33 + Lp[1]       
        Dl0=-Lp[0] * (4) +Lp[1]
        
        gamd=np.zeros(temp.shape)
        dum=temp>=4
        gamd[dum]=-np.log(Sr[dum] / 100) / Dl[dum]
        gamd[~dum]=-np.log(Sr0/100)/Dl0
        
        return Dl, Sr, gamd 
        
    # Dl, Sr, gamd =CUL_PIP_PIP_Pup_dev_mor(TT)
    # plt.plot(TT,-np.log(Sr / 100) / Dl)
    # plt.plot(TT,gamd)    

def CUL_PIP_PIP_vectorcompetence(temp):
        BC=np.zeros(temp.shape)    
        dum=(temp>=16.9) & (temp<=38.8)
        Qp=np.array([ 3e-3,   16.8,   38.9])
        BC[dum] = -Qp[0] * (temp[dum] - Qp[1]) * (temp[dum] - Qp[2])
        BC[~dum]=0.001
        
        return BC         



def calculate_rates_PIP_MOL(TT,NumNMS):
        rates=np.zeros((18,TT.size))
        
        Dl,Sr,gamd =CUL_PIP_MOL_egg_dev_mor(TT)
        rates[0, :] = NumNMS / Dl
        rates[1, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_MOL_L1_dev_mor(TT)
        rates[2, :] = NumNMS / Dl
        rates[3, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_MOL_L2_dev_mor(TT)
        rates[4, :] = NumNMS / Dl
        rates[5, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_MOL_L3_dev_mor(TT)
        rates[6, :] = NumNMS / Dl
        rates[7, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_MOL_L4_dev_mor(TT)
        rates[8, :] = NumNMS / Dl
        rates[9, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_MOL_Pup_dev_mor(TT)
        rates[10, :] = NumNMS / Dl
        rates[11, :]= gamd
        
        Dl=CUL_PIP_MOL_Adl_longe(TT)
        rates[12, :] = 5 / Dl
        rates[13, :]= 1 / Dl
        
        Dl_inv=CUL_PIP_PIP_gonc(TT)
        rates[14, :] = NumNMS * Dl_inv
        rates[15, :]= Dl_inv
        
        ov=CUL_PIP_PIP_ov(TT)
        rates[16, :]= ov
        
        BC=CUL_PIP_PIP_vectorcompetence(TT)
        rates[17, :]= BC      

        return rates

def calculate_rates_PIP_PIP(TT,NumNMS):
        rates=np.zeros((19,TT.size))
        
        Dl,Sr,gamd =CUL_PIP_MOL_egg_dev_mor(TT)
        rates[0, :] = NumNMS / Dl
        rates[1, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_PIP_L1_dev_mor(TT)
        rates[2, :] = NumNMS / Dl
        rates[3, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_PIP_L2_dev_mor(TT)
        rates[4, :] = NumNMS / Dl
        rates[5, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_PIP_L3_dev_mor(TT)
        rates[6, :] = NumNMS / Dl
        rates[7, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_PIP_L4_dev_mor(TT)
        rates[8, :] = NumNMS / Dl
        rates[9, :]= gamd
        
        Dl,Sr,gamd =CUL_PIP_PIP_Pup_dev_mor(TT)
        rates[10, :] = NumNMS / Dl
        rates[11, :]= gamd
        
        Dl=CUL_PIP_PIP_Adl_longe(TT)
        rates[12, :] = 5 / Dl
        rates[13, :]= 1 / Dl
        
        Dl_inv=CUL_PIP_PIP_gonc(TT)
        rates[14, :] = NumNMS * Dl_inv
        rates[15, :]= Dl_inv
        
        ov=CUL_PIP_PIP_ov(TT)
        rates[16, :]= ov
        
        BC=CUL_PIP_PIP_vectorcompetence(TT)
        rates[17, :]= BC  
        
        Dl=CUL_PIP_PIP_Adl_longe_di(TT)
        rates[18, :] = 5 / Dl
        
        return rates


