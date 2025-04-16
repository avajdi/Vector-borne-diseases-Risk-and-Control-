
"""
Created on Tue Apr  8 08:08:49 2025

@author: avajdi
"""




import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import os
from cord_to_CRofland import cord_to_CRofland
from MosquitoRates import calculate_rates_PIP_MOL
from Dif_functions import  prepare_inputs2 , odeforward_PIP_mh , odebackward_PIP_mh , dfdci_PIP_mh
def Optimize_PIP_MOL(
        longitude='-74.28' ,
        latitude='40.8' , 
        len_ins='2' ,   
        len_lr='20'  ,     
        len_cr='40' ,      
        numtrls='2'  , 
        ls_ef='0.8'  ,   
        numtris='2' ,    
        is_ef='0.2'  , 
        numtrcl='2' ,   
        cl_ef='0.5'  ,  
        numtrls_mh='1' ,  
        ls_ef_mh='0.8' ,    
        numtris_mh='1'  ,   
        is_ef_mh='0.2'  , 
        numtrcl_mh='1' ,   
        cl_ef_mh='0.5',  
        adtemp='temp_2023_daily_average_py.mat',
        adpre='pre_pastyearsav_py.mat' 
):
    
    
    
    
    # longitude='-74.28' 
    # latitude='40.8'  
    # len_ins='2'    
    # len_lr='20'       
    # len_cr='40'       
    # numtrls='2'   
    # ls_ef='0.8'     
    # numtris='2'     
    # is_ef='0.2'   
    # numtrcl='2'    
    # cl_ef='0.5'    
    # numtrls_mh='1'   
    # ls_ef_mh='0.8'     
    # numtris_mh='1'     
    # is_ef_mh='0.2'   
    # numtrcl_mh='1'    
    # cl_ef_mh='0.5'
    # #adtemp='temp_pastyearsav_py.mat'   
    # adtemp='temp_2023_daily_average_py.mat'
    # adpre='pre_pastyearsav_py.mat' 
    
    
    
    
    # Convert inputs to appropriate types
    longitude = float(longitude)
    latitude = float(latitude)
    len_ins = float(len_ins)
    len_lr = float(len_lr)
    len_cr = float(len_cr)
    numtrls = int(numtrls)
    ls_ef = float(ls_ef)
    numtris = int(numtris)
    is_ef = float(is_ef)
    numtrcl = int(numtrcl)
    cl_ef = float(cl_ef)
    numtrls_mh = int(numtrls_mh)
    ls_ef_mh = float(ls_ef_mh)
    numtris_mh = int(numtris_mh)
    is_ef_mh = float(is_ef_mh)
    numtrcl_mh = int(numtrcl_mh)
    cl_ef_mh = float(cl_ef_mh)

    col, row = cord_to_CRofland(longitude, latitude)
    # Load input data from .mat files
    temp_data = loadmat(adtemp) 
    temp=(np.squeeze(temp_data['dailyaverage'][row,col,range(temp_data['indexend'][0][0])]).astype(float) /
         (np.squeeze(temp_data['scaling'][0][0]).astype(float))-273.15)
    temp = np.tile(temp, 3)  # Repeat for three years
    temp_ma=pd.Series(temp).rolling(window=3, min_periods=1).mean().values
    temp_mh=temp.copy()
    dum1=temp<5
    dum2=temp>30
    temp_mh[dum1]=5.0
    temp_mh[dum2]=30.0
    pre_data = loadmat(adpre)   
    pre=(np.squeeze(pre_data['dailytot'][row,col,range(pre_data['indexend'][0][0])]).astype(float) /
     (np.squeeze(pre_data['scaling']).astype(float)))
    pre = np.tile(pre, 3)
    pre_ma = pd.Series(pre).rolling(window=15, min_periods=1).mean().values
    
    del temp_data, pre_data, row, col, pre
   

    # Date calculations
    today = datetime.today()
    start_date = datetime(today.year, 1, 1)
    dates = np.array([start_date - timedelta(days=i) for i in range(730, 0, -1)] +
                     [start_date + timedelta(days=i) for i in range(365)])

    del start_date, today 
  

    dates = dates[129:]
    temps = np.tile(temp[129:], (10, 1))
    temps_ma = np.tile(temp_ma[129:], (10, 1))
    temps_mh = np.tile(temp_mh[129:], (10, 1))
    pres = np.tile(pre_ma[129:], (10, 1))
    temps = np.vstack((temps, temps_mh))
    del temp, pre_ma ,temp_mh , temps_mh , temp_ma
    
    # Profiles for larval and adult stages control measures
    del_sprofil = 0.01
    sprofil_ins = np.concatenate((np.zeros(1), np.arange(0, 1 + del_sprofil, del_sprofil),
                                  np.ones(int(len_ins / del_sprofil)), np.arange(1, 0, -del_sprofil),
                                  np.zeros(int(len(dates) / del_sprofil))))
    f1tu=np.concatenate((np.zeros(1), sprofil_ins[0:-3]))
    f_1tu=sprofil_ins[1:-1]
    drsprofil_ins = (f1tu-f_1tu)/(2*del_sprofil)
    
    sprofil_lr = np.concatenate((np.zeros(1), np.arange(0, 1 + del_sprofil, del_sprofil),
                                 np.ones(int(len_lr / del_sprofil)), np.arange(1, 0, -del_sprofil),
                                 np.zeros(int(len(dates) / del_sprofil))))

    f1tu=np.concatenate((np.zeros(1), sprofil_lr[0:-3]))
    f_1tu=sprofil_lr[1:-1]
    drsprofil_lr = (f1tu-f_1tu)/(2*del_sprofil)

    # Delta profile for larval dynamics
    del_delta = 0.01
    tau = np.concatenate((np.arange(0, 4 + del_delta, del_delta),
                         4+ np.arange(del_delta, len(dates),del_delta)))

    sigma = np.concatenate((np.zeros(1), np.arange(0, 2 + del_delta, del_delta),
                            np.arange(2 - del_delta, 0, -del_delta),
                            np.zeros(len(np.arange(del_delta, len(dates),del_delta))))) / 4

    eta = 1 / len_cr
    dum = sigma * np.exp(eta * tau)
    int_profile = np.concatenate((np.zeros(1),np.cumsum(dum[:-1] + dum[1:]) * del_delta / 2))
    deltaprofil = int_profile * np.exp(-eta * tau)
    f1tu=np.concatenate((np.zeros(1), deltaprofil[0:-3]))
    f_1tu=deltaprofil[1:-1]
    drdeltaprofil =(f1tu-f_1tu)/(2*del_delta)

   

    del f1tu, f_1tu, dum, tau, int_profile, sigma

    Nhr=10000000
    R=temps.shape[0]

    dum1=pres>0.0084
    dum2=pres<0.0038
    dum3=~(dum1|dum2)
    Kt=(dum1*1.6+dum3*1.2+dum2*1) 
    Kt[:,:]=1
    Kt_mh=np.ones(Kt.shape)
 
    Cl0t=0.06*np.ones(Kt.shape)*Nhr
    Cl0t_mh=0.005*np.ones(Kt.shape)*Nhr
    
    Cl0t=np.vstack((Cl0t,Cl0t_mh))
    Kt=np.vstack((Kt,Kt_mh))
    del dum1, dum2, dum3, Cl0t_mh, Kt_mh
    
    
    
    numtr_ls=2
    lendat=len(dates)
    dum=np.floor((lendat-20)/numtr_ls)-10
    lstim=np.random.randint(1, dum + 1, size=(numtr_ls, R))+10 
    lstim = np.cumsum(lstim, axis=0)
    lstim = lstim.astype(float)
    lslev=np.zeros(lstim.shape)
    
    
    numtr_is=2
    dum=np.floor((lendat-20)/numtr_is)-10
    istim=np.random.randint(1, dum + 1, size=(numtr_is, R))+10  
    istim=np.cumsum(istim, axis=0)
    istim = istim.astype(float)
    islev=np.zeros(istim.shape)
    
    

    numtr_cl=1
    dum=np.floor((lendat-20)/numtr_cl)-10
    taucl=np.random.randint(1, dum + 1, size=(numtr_cl, R))+10  
    taucl=np.cumsum(taucl, axis=0)
    taucl= taucl.astype(float)
    alpha=np.zeros(taucl.shape)
    
    
    
    
    inps = {
        "lstim": lstim, "lslev": lslev,
        "istim": istim, "islev": islev,
        "taucl": taucl, "alpha": alpha,
        "sprofil_lr": sprofil_lr, "drsprofil_lr": drsprofil_lr,
        "sprofil_ins": sprofil_ins, "drsprofil_ins": drsprofil_ins,
        "del_sprofil": del_sprofil,
        "deltaprofil": deltaprofil, "drdeltaprofil": drdeltaprofil,
        "del_delta": del_delta
    }
   
    
    inps.update({"temps": temps, "temps_ma": temps_ma, "Nhr": Nhr, "R": R, "Cl0t": Cl0t, "Kt": Kt})
    numNMst=50;
    inps['eind'] = np.arange(0, numNMst)  # 1 to 50
    inps['l1ind'] = np.arange(inps['eind'][-1] + 1, inps['eind'][-1] + numNMst+1)  # Last of eind + 1 to Last of eind + 50
    inps['l2ind'] = np.arange(inps['l1ind'][-1] + 1, inps['l1ind'][-1] + numNMst+1)
    inps['l3ind'] = np.arange(inps['l2ind'][-1] + 1, inps['l2ind'][-1] + numNMst+1)
    inps['l4ind'] = np.arange(inps['l3ind'][-1] + 1, inps['l3ind'][-1] + numNMst+1)
    inps['pind'] = np.arange(inps['l4ind'][-1] + 1, inps['l4ind'][-1] + numNMst+1)
    inps['a1ind'] = np.arange(inps['pind'][-1] + 1, inps['pind'][-1] + numNMst+1)
    inps['a2ind'] = np.arange(inps['a1ind'][-1] + 1, inps['a1ind'][-1] + numNMst+1)
    inps['a3ind'] = np.arange(inps['a2ind'][-1] + 1, inps['a2ind'][-1] + numNMst+1)
    inps['a4ind'] = np.arange(inps['a3ind'][-1] + 1, inps['a3ind'][-1] + numNMst+1)
    inps['a5ind'] = np.arange(inps['a4ind'][-1] + 1, inps['a4ind'][-1] + numNMst+1)

    
    inps['linds']=np.concatenate([inps['l1ind'] , inps['l2ind'] , inps['l3ind'] , inps['l4ind']])    
    inps['ainds']=np.hstack([inps['a1ind'] , inps['a2ind'] , inps['a3ind'] , inps['a4ind'], inps['a5ind']])
    inps['a2to5inds']=np.hstack([ inps['a2ind'] , inps['a3ind'] , inps['a4ind'], inps['a5ind']])
    inps['a1to4inds']=np.hstack([inps['a1ind'] , inps['a2ind'] , inps['a3ind'] , inps['a4ind']])
    inps['aindsfirst']=np.hstack([inps['a1ind'][0] , inps['a2ind'][0] , inps['a3ind'][0] , inps['a4ind'][0], inps['a5ind'][0]])
    inps['aindslast']=np.hstack([inps['a1ind'][-1] , inps['a2ind'][-1] , inps['a3ind'][-1] , inps['a4ind'][-1], inps['a5ind'][-1]])
    inps['Shainds']=np.hstack([inps['a1ind'][-1] , inps['a1ind'][:-1],
                               inps['a2ind'][-1] , inps['a2ind'][:-1],
                               inps['a3ind'][-1] , inps['a3ind'][:-1],
                               inps['a4ind'][-1] , inps['a4ind'][:-1],
                               inps['a5ind'][-1] , inps['a5ind'][:-1]])
    
    inps['Shainds_bk']=np.hstack([inps['a1ind'][1:] , inps['a1ind'][0],
                                  inps['a2ind'][1:] , inps['a2ind'][0],
                                  inps['a3ind'][1:] , inps['a3ind'][0],
                                  inps['a4ind'][1:] , inps['a4ind'][0],
                                  inps['a5ind'][1:] , inps['a5ind'][0]])
    
    TT=np.arange(-70,70.5,0.5)
    inps.update({'rates':calculate_rates_PIP_MOL(TT,numNMst)})
    inps.update({"temp_to_mh_mn": 5.0, "temp_out_mh_mn": 5.0, "temp_to_mh_mx": 33.0, "temp_out_mh_mx": 33.0, "fto": 2.0 , "fout": 2.0})
    
    
    # Initial conditions and solver setup
    y0=np.zeros((R,11*numNMst))
    y0[np.arange(int(np.round(R/2))),]=(10/50)*(Nhr/1000)
    y0=y0.ravel()  
  
    tspan = np.arange(0.1, len(dates), 0.1)
    options = {'rtol': 1e-4}

    #inps0, inps1, inps2 = prepare_inputs(inps)
    inps0i, inps0f, inps1i, inps1f, inps2i, inps2f = prepare_inputs2(inps)
    start_time = time.time()
    solution = solve_ivp(odeforward_PIP_mh, [tspan[0], tspan[-1]], y0,method='RK45', t_eval=tspan, args=(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f),**options)
    end_time = time.time()
    end_time - start_time
    t=solution.t
    y=solution.y
    #dum=np.arange(len(dates)-1)
    # y2=np.sum(y[np.ix_(inps['ainds'],dum*10)],axis=0)
    #y2=np.sum(y[np.ix_(inps['ainds']+11*550,dum*10)],axis=0)
    # plt.plot(dates[dum],y2)
    #dum2=calculate_rates_PIP_MOL(inps['temps'][0,dum],numNMst)
    #risk=(((dum2[15,]/dum2[13,])**2)*dum2[17,])*(0.1/(0.1+dum2[13,]))*y2/Nhr   
    #plt.plot(dates[dum],pd.Series(risk).rolling(window=11, center=True, min_periods=1).mean().values)
#####################################################################################################################
    lendat = len(dates)
    stdum = len(dates) - 370
    endum = lendat

    dates = dates[stdum:endum]
    lendat = len(dates)
    y0=np.ascontiguousarray(y[:,int(np.where(np.isclose(t, stdum+0.1, atol=1e-6))[0][0])])
    tspan = np.round(np.arange(0.2, lendat - 0.1, 0.1),1)
    tind=np.floor(tspan).astype(np.int32)
    temps = temps[:, stdum:endum]
    temps_ma = temps_ma[:, stdum:endum]
    pres = pres[:, stdum:endum]
    Cl0t=Cl0t[:, stdum:endum]
    Kt=Kt[:, stdum:endum]
    inps.update(Cl0t=Cl0t, Kt=Kt, temps=temps, temps_ma=temps_ma )
    
    #######################
    Nep=80
    RR=round(R/2)
    numtr_ls=np.max([numtrls,numtrls_mh])
    dum=np.floor((lendat-20)/numtr_ls)-10
    lstim=np.random.randint(1, dum + 1, size=(numtr_ls, R))+10 
    lstim = np.cumsum(lstim, axis=0)
    lstim = lstim.astype(float)
    lslev=np.zeros(lstim.shape)
    lslev[0:numtrls,0:RR]=ls_ef
    lslev[0:numtrls_mh,RR:R]=ls_ef_mh
    
    
    numtr_is=np.max([numtris,numtris_mh])
    dum=np.floor((lendat-20)/numtr_is)-10
    istim=np.random.randint(1, dum + 1, size=(numtr_is, R))+10  
    istim=np.cumsum(istim, axis=0)
    istim = istim.astype(float)
    islev=np.zeros(istim.shape)
    islev[0:numtris,0:RR]=is_ef
    islev[0:numtris_mh,RR:R]=is_ef_mh
    


    numtr_cl=np.max([numtrcl,numtrcl_mh])
    dum=np.floor((lendat-20)/numtr_cl)-10
    taucl=np.random.randint(1, dum + 1, size=(numtr_cl, R))+10  
    taucl=np.cumsum(taucl, axis=0)
    taucl= taucl.astype(float)
    alpha1=np.mean(Cl0t[0,:])/Nhr*cl_ef
    alpha2=np.mean(Cl0t[RR,:])/Nhr*cl_ef_mh
    alpha=np.zeros(taucl.shape)
    alpha[0:numtrcl,0:RR]=alpha1
    alpha[0:numtrcl_mh,RR:R]=alpha2
    
    
    
    
    lstimep = np.zeros((numtr_ls, R, Nep+1))
    lstimep[:, :, 0] = lstim
    istimep = np.zeros((numtr_is, R, Nep+1))
    istimep[:, :, 0] = istim
    tauclep = np.zeros((numtr_cl, R, Nep+1))
    tauclep[:, :, 0] = taucl

    valtomin = np.zeros((Nep+1, R))

# Initialize dFdt arrays as 2D arrays
    dFdtls = np.zeros(( numtr_ls,R))
    dFdtlsmov = np.zeros((numtr_ls,R))
    dFdtlsSQmov = np.zeros(( numtr_ls,R))

    dFdtis = np.zeros(( numtr_is,R))
    dFdtismov = np.zeros(( numtr_is,R))
    dFdtisSQmov = np.zeros(( numtr_is,R))
    

    dFdtau = np.zeros(( numtr_cl,R))
    dFdtaumov = np.zeros(( numtr_cl,R))
    dFdtauSQmov = np.zeros(( numtr_cl,R))
    
    
    inps['a'] = np.zeros(( len(tspan),R))
    inps['l'] = np.zeros(( len(tspan),R))
    inps['en'] = np.zeros(( len(tspan),R))
    
    for ep in range(Nep):
        start_time = time.time()
        
        inps['lstim'] = lstimep[:, :, ep]
        inps['istim'] = istimep[:, :, ep]
        inps['lslev'] = lslev
        inps['islev'] = islev
        inps['taucl'] = tauclep[:, :, ep]
        inps['alpha'] = alpha
        
        
        inps0i, inps0f, inps1i, inps1f, inps2i, inps2f = prepare_inputs2(inps)
       # start_time = time.time()
        solution = solve_ivp(odeforward_PIP_mh, [tspan[0], tspan[-1]], y0,method='RK45', t_eval=tspan, args=(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f),**options)
       # end_time = time.time()
        #end_time - start_time
        t=solution.t
        y=solution.y
        # y2=np.sum(y[inps['ainds'],],axis=0)       
        # plt.plot(t,y2)
        
        mxin=round(y.shape[0]/R)
        for rr in range(R):
            ofs=rr*mxin
            inps['a'][:,rr] = np.sum(y[inps['ainds']+ofs,],axis=0) 
            inps['l'][:,rr] = np.sum(y[inps['linds']+ofs,],axis=0)
            inps['en'][:,rr] = np.sum(y[inps['eind'][-1:]+ofs,],axis=0)
            

        inps['weis']=np.ones(R)
        inps['t0el']=tspan[0]
        inps['deltel']=np.round(tspan[1]-tspan[0],1)
        tauspan=np.round(lendat-tspan[::-1],1)
        tauspan[0]=tauspan[0]+0.01
        tauspan[-1]=tauspan[-1]-0.01
        inps['ff']=float(lendat)
        inps['mode']=1 

        
        inps0i, inps0f, inps1i, inps1f, inps2i, inps2f = prepare_inputs2(inps)
        lam0=np.zeros(y0.shape)
    
        #start_time = time.time()
        solution = solve_ivp(odebackward_PIP_mh, [tauspan[0], tauspan[-1]], lam0,method='RK45', t_eval=tauspan, args=(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f),**options)
        #end_time = time.time()
        #end_time - start_time
        tau=solution.t
        lam=solution.y
        lam=lam/Nhr
        
            # rr=7
            # ofs=rr*mxin
            # lam_a = lam[inps['ainds']+ofs,]           
            # lam_lr = lam[inps['linds']+ofs,]            
            # lam_e = lam[inps['eind']+ofs,]
            # plt.plot(lam_e.T)
            
        Tm=(inps['temps'] [:,np.floor(t).astype(int)]).T
        dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
        ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
        dum=dum-ind
         
        gamadRisk=inps['rates'][13,ind]*(1-dum)+inps['rates'][13,ind+1]*(dum)   
        gamaeRisk=inps['rates'][15,ind]*(1-dum)+inps['rates'][15,ind+1]*(dum)  
        BC=inps['rates'][17,ind]*(1-dum)+inps['rates'][17,ind+1]*(dum)
        risk=((1/(1+10*gamadRisk)) *BC*(gamaeRisk/gamadRisk)**2)*(inps['a']/Nhr)
        valtomin[ep,:]=np.mean(risk,axis=0) 
        
     
        
        dfdldrt,dfdadrt, dfdalph_kt = dfdci_PIP_mh(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f,lam,y,tspan,mxin)
        
        
        # plt.plot(dfdalph_kt.T)
        # plt.plot(dfdadrt.T)
        # plt.plot(dfdldrt.T)
    
        
        
        
        for num in range(numtr_ls):
            tuc=lstimep[num, :, ep]
            tuc=tuc[:,None]  
            inrw = ((tspan-tuc)/del_sprofil)
            ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
            dum = np.maximum(0, inrw - ind )
            dum2 = ( (1 - dum) * drsprofil_lr[ind] + dum * drsprofil_lr[ind+1] ) * dfdldrt
            dFdtls[num,:] =  np.sum(dum2[:,1:]+dum2[:,:-1],axis=1) * (inps['deltel'] * lslev[num,] * 0.5)    
        
        
         
        for num in range(numtr_is):
            tuc=istimep[num, :, ep]
            tuc=tuc[:,None]  
            inrw = ((tspan-tuc)/del_sprofil)
            ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
            dum = np.maximum(0, inrw - ind )
            dum2 = ( (1 - dum) * drsprofil_ins[ind] + dum * drsprofil_ins[ind+1] ) * dfdadrt
            dFdtis[num,:] =  np.sum(dum2[:,1:]+dum2[:,:-1],axis=1) * (inps['deltel'] * islev[num,] * 0.5)    
        
        
        
        
        for num in range(numtr_cl):
            tuc=tauclep[num, :, ep]
            tuc=tuc[:,None]  
            inrw = ((tspan-tuc)/del_delta)
            ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
            dum = np.maximum(0, inrw - ind )
            dum2 = - ( (1 - dum) * drdeltaprofil[ind] + dum * drdeltaprofil[ind+1] ) * dfdalph_kt
            dFdtau[num,:] =  np.sum(dum2[:,1:]+dum2[:,:-1],axis=1) * (inps['deltel'] * alpha[num,] * Nhr * 0.5)        
        
        
        
     ##########################################################   
         
        if ep<100:
            fr1=(ep)/(ep+1)
            fr2=fr1
        else:
            fr1=0.99
            fr2=0.99

       
        dFdtlsmov=fr1*dFdtlsmov+(1-fr1)*dFdtls
        dFdtlsSQmov=fr2*dFdtlsSQmov+(1-fr2)*(dFdtls)**2
        
        dFdtismov=fr1*dFdtismov+(1-fr1)*dFdtis
        dFdtisSQmov=fr2*dFdtisSQmov+(1-fr2)*(dFdtis)**2
        
        
        dFdtaumov=fr1*dFdtaumov+(1-fr1)*dFdtau
        dFdtauSQmov=fr2*dFdtauSQmov+(1-fr2)*(dFdtau)**2
        
        lr=1;
        
        delttls=lr*dFdtlsmov / (np.sqrt(dFdtlsSQmov)+1e-12)
        delttis=lr*dFdtismov / (np.sqrt(dFdtisSQmov)+1e-12)
        delttau=lr*dFdtaumov / (np.sqrt(dFdtauSQmov)+1e-12)
        
        lstimep[:,:,ep+1]=np.minimum(lendat-20,np.maximum(lstimep[:,:,ep]-delttls,10))
        istimep[:,:,ep+1]=np.minimum(lendat-20,np.maximum(istimep[:,:,ep]-delttis,10))
        tauclep[:,:,ep+1]=np.minimum(lendat-20,np.maximum(tauclep[:,:,ep]-delttau,10))
                   

        
        end_time = time.time()
        
        print(f"time and epoch : {end_time - start_time} and {ep}")
        print(f"valtomin: {np.mean(valtomin[ep,:])}")
        
        
        ########################################################################################
        
     
    arr=valtomin[0:-1, 0:RR]+valtomin[0:-1,RR:R]
    min_value = np.min(arr)
    min_index = np.unravel_index(np.argmin(arr), arr.shape)
    
    inps['lstim'] = lstimep[:, :, min_index[0]]
    inps['istim'] = istimep[:, :, min_index[0]]
    inps['taucl'] = tauclep[:, :, min_index[0]]
    inps['lslev'] = 0.0 *lslev
    inps['islev'] = 0.0 * islev
    inps['alpha'] = 0.0 * alpha      
    inps0i, inps0f, inps1i, inps1f, inps2i, inps2f = prepare_inputs2(inps)
    solution = solve_ivp(odeforward_PIP_mh, [tspan[0], tspan[-1]], y0,method='RK45', t_eval=tspan, args=(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f),**options)
    t=solution.t
    y=solution.y
    # y2=np.sum(y[inps['ainds'],],axis=0)       
    # plt.plot(t,y2)  
    mxin=round(y.shape[0]/R)
    ofs1=min_index[1]*mxin
    ofs2=(min_index[1]+RR)*mxin
    initial={}
    initial['a'] = np.sum(y[inps['ainds']+ofs1,],axis=0) 
    initial['a_mh'] = np.sum(y[inps['ainds']+ofs2,],axis=0)
    initial['l'] = np.vstack((np.sum(y[inps['linds']+ofs1,],axis=0),np.sum(y[inps['linds']+ofs2,],axis=0))).T
    initial['e'] = np.vstack((np.sum(y[inps['eind']+ofs1,],axis=0), np.sum(y[inps['eind']+ofs2,],axis=0))).T
        
    Tm=inps['temps'] [np.ix_((min_index[1],min_index[1]+RR),np.floor(t).astype(int))]
    dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
    ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
    dum=dum-ind         
    gamadRisk=inps['rates'][13,ind]*(1-dum)+inps['rates'][13,ind+1]*(dum)   
    gamaeRisk=inps['rates'][15,ind]*(1-dum)+inps['rates'][15,ind+1]*(dum)  
    BC=inps['rates'][17,ind]*(1-dum)+inps['rates'][17,ind+1]*(dum)
    initial['risk'] =np.sum(((1/(1+10*gamadRisk)) *BC*(gamaeRisk/gamadRisk)**2)*(np.vstack((initial['a'],initial['a_mh']))/Nhr),axis=0)

        
        
    
    inps['lslev'] = lslev
    inps['islev'] = islev
    inps['alpha'] = alpha         
    inps0i, inps0f, inps1i, inps1f, inps2i, inps2f = prepare_inputs2(inps)
    solution = solve_ivp(odeforward_PIP_mh, [tspan[0], tspan[-1]], y0,method='RK45', t_eval=tspan, args=(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f),**options)
    y=solution.y

    final={}
    final['a'] = np.sum(y[inps['ainds']+ofs1,],axis=0) 
    final['a_mh'] =  np.sum(y[inps['ainds']+ofs2,],axis=0)
    final['l'] = np.vstack((np.sum(y[inps['linds']+ofs1,],axis=0),np.sum(y[inps['linds']+ofs2,],axis=0))).T
    final['e'] = np.vstack((np.sum(y[inps['eind']+ofs1,],axis=0), np.sum(y[inps['eind']+ofs2,],axis=0))).T      
    final['risk'] =np.sum(((1/(1+10*gamadRisk)) *BC*(gamaeRisk/gamadRisk)**2)*(np.vstack((final['a'],final['a_mh']))/Nhr),axis=0)
        

        
    datels=np.sort(dates[np.round(lstimep[0:numtrls, min_index[1], min_index[0]]).astype(np.int32)])
    datels_mh=np.sort(dates[np.round(lstimep[0:numtrls_mh, min_index[1]+RR, min_index[0]]).astype(np.int32)])
    dateis=np.sort(dates[np.round(istimep[0:numtris, min_index[1], min_index[0]]).astype(np.int32)])
    dateis_mh=np.sort(dates[np.round(istimep[0:numtris_mh, min_index[1]+RR, min_index[0]]).astype(np.int32)])
    datetau=np.sort(dates[np.round( tauclep[0:numtrcl, min_index[1], min_index[0]]).astype(np.int32)])
    datetau_mh=np.sort(dates[np.round( tauclep[0:numtrcl_mh, min_index[1]+RR, min_index[0]]).astype(np.int32)])    
        
        
    inddtints = np.searchsorted(tspan, np.arange(1,len(dates),1))
    ain=pd.Series(initial['a']).rolling(window=7, center=True).mean()
    a_mhin=pd.Series(initial['a_mh']).rolling(window=7, center=True).mean()
    lin=pd.Series(initial['l'][:,0]).rolling(window=7, center=True).mean()
    l_mhin=pd.Series(initial['l'][:,1]).rolling(window=7, center=True).mean()
    ein=pd.Series(initial['e'][:,0]).rolling(window=7, center=True).mean()
    e_mhin=pd.Series(initial['e'][:,1]).rolling(window=7, center=True).mean()
    riskin=pd.Series(initial['risk']).rolling(window=7, center=True).mean()
    
    af=pd.Series(final['a']).rolling(window=7, center=True).mean()
    a_mhf=pd.Series(final['a_mh']).rolling(window=7, center=True).mean()
    lf=pd.Series(final['l'][:,0]).rolling(window=7, center=True).mean()
    l_mhf=pd.Series(final['l'][:,1]).rolling(window=7, center=True).mean()
    ef=pd.Series(final['e'][:,0]).rolling(window=7, center=True).mean()
    e_mhf=pd.Series(final['e'][:,1]).rolling(window=7, center=True).mean()
    riskf=pd.Series(final['risk']).rolling(window=7, center=True).mean()  
        
           # plt.plot(dates[np.arange(1,len(dates),1)],lf[inddtints])
           # plt.plot(dates[np.arange(1,len(dates),1)],l_mhf[inddtints])
           # plt.figure()
           # plt.plot(dates[np.arange(1,len(dates),1)],riskin[inddtints])
           # plt.plot(dates[np.arange(1,len(dates),1)],riskf[inddtints])
        
    pref="pipien_mol"  
    os.makedirs(pref, exist_ok=True)
        
    out = riskf[inddtints]
    np.savetxt(os.path.join(pref, 'riskf.txt'), out, delimiter=' ')
    out = riskin[inddtints]
    np.savetxt(os.path.join(pref, 'riskin.txt'), out, delimiter=' ')
    
    out = af[inddtints]
    np.savetxt(os.path.join(pref, 'af.txt'), out, delimiter=' ')
    out = ain[inddtints]
    np.savetxt(os.path.join(pref, 'ain.txt'), out, delimiter=' ')
    
    out = a_mhf[inddtints]
    np.savetxt(os.path.join(pref, 'a_mhf.txt'), out, delimiter=' ')
    out = a_mhin[inddtints]
    np.savetxt(os.path.join(pref, 'a_mhin.txt'), out, delimiter=' ')  
    
    out = lf[inddtints]
    np.savetxt(os.path.join(pref, 'lf.txt'), out, delimiter=' ')
    out = lin[inddtints]
    np.savetxt(os.path.join(pref, 'lin.txt'), out, delimiter=' ')
    
    out = l_mhf[inddtints]
    np.savetxt(os.path.join(pref, 'l_mhf.txt'), out, delimiter=' ')
    out = l_mhin[inddtints]
    np.savetxt(os.path.join(pref, 'l_mhin.txt'), out, delimiter=' ')
    
    out=(dates[np.arange(1,len(dates),1)]).astype('datetime64[D]')
    #     # pd.Series(out).to_csv('dates.txt', index=False,header=False)
    np.savetxt(os.path.join(pref, 'dates.txt'), out.astype(str), fmt='%s', delimiter=' ') 
        
    out=datels.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'datels.txt'), out.astype(str), fmt='%s', delimiter=' ')
    out=dateis.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'dateis.txt'), out.astype(str), fmt='%s', delimiter=' ')
    out=datetau.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'datetau.txt'), out.astype(str), fmt='%s', delimiter=' ')
    
    out=datels_mh.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'datels_mh.txt'), out.astype(str), fmt='%s', delimiter=' ')
    out=dateis_mh.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'dateis_mh.txt'), out.astype(str), fmt='%s', delimiter=' ')
    out=datetau_mh.astype('datetime64[D]')
    np.savetxt(os.path.join(pref, 'datetau_mh.txt'), out.astype(str), fmt='%s', delimiter=' ')
  #  return 0
        
        
        