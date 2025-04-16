

import numpy as np
from numba import types
from numba.typed import Dict
from numba import njit


def prepare_inputs(inps):
    """
    Converts a regular Python dictionary to Numba-compatible dictionaries
    for scalars, 1D arrays, and 2D arrays.
    """
    # Create separate Numba-typed dictionaries
    numba_scalars = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    numba_1d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    numba_2d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:, :])

    for key, value in inps.items():
        if np.isscalar(value):
            # If the value is a scalar, store it in the scalars dictionary
            numba_scalars[key] = float(value)
        elif value.ndim == 1:
            # If it's a 1D array, store it in the 1D arrays dictionary
            numba_1d_arrays[key] = np.array(value, dtype=np.float64)
        elif value.ndim == 2:
            # If it's a 2D array, store it in the 2D arrays dictionary
            numba_2d_arrays[key] = np.array(value, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported input format for key '{key}'")
    
    return numba_scalars, numba_1d_arrays, numba_2d_arrays


def prepare_inputs2(inps):
    """
    Converts a regular Python dictionary to Numba-compatible dictionaries
    for scalars, 1D arrays, and 2D arrays, preserving the original data types.
    """
    # Create separate Numba-typed dictionaries for int and float types
    numba_int_scalars = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
    numba_float_scalars = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    numba_int_1d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.int32[:])
    numba_float_1d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

    numba_int_2d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.int32[:, :])
    numba_float_2d_arrays = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:, :])

    for key, value in inps.items():
        if np.isscalar(value):
            # Determine type and store the value in the corresponding dictionary
            if isinstance(value, np.float64) or isinstance(value, float):
                numba_float_scalars[key] = value
            elif isinstance(value, np.int32) or isinstance(value, int):
                numba_int_scalars[key] = value
        elif hasattr(value, 'ndim'):
            # Determine type and dimension to store the value in the correct dictionary
            if value.dtype == np.int32:
                if value.ndim == 1:
                    numba_int_1d_arrays[key] = np.ascontiguousarray(value)
                elif value.ndim == 2:
                    numba_int_2d_arrays[key] = np.ascontiguousarray(value)
            elif value.dtype == np.float64 or value.dtype == float:
                if value.ndim == 1:
                    numba_float_1d_arrays[key] = np.ascontiguousarray(value)
                elif value.ndim == 2:
                    numba_float_2d_arrays[key] = np.ascontiguousarray(value)
        else:
            raise ValueError(f"Unsupported input format for key '{key}'")
    
    return  numba_int_scalars, numba_float_scalars, numba_int_1d_arrays, numba_float_1d_arrays, numba_int_2d_arrays, numba_float_2d_arrays,
    



# JIT-compiled function
@njit
def odeforward_PIP_di(t, y, inps0i,inps0f,inps1i,inps1f,inps2i,inps2f):
  

    R = int(inps0i['R'])
    flrt=int(np.floor(t))
   
    Kt = inps2f['Kt'][:, flrt]
    
    Cl0t = inps2f['Cl0t']
    if Cl0t.shape[1] > 1:
        Cl0t = Cl0t[:, flrt ]
    else:
        Cl0t = Cl0t[:, 0 ]

    del_delta = inps0f['del_delta']
    
    taucl = inps2f['taucl']
    shsh=taucl.shape
    taucl=np.ascontiguousarray(taucl).ravel()
    inrw = (t - taucl) / del_delta
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    
    Clt = inps0i['Nhr'] * inps0f['alpha'] * (
        (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
    )
    Clt=Clt.reshape(shsh)
    Clt = np.sum(Clt, axis=0)
    Clt = np.maximum(Cl0t - Clt, 0) * Kt
    
    
  
   
    del_sprofil = inps0f['del_sprofil']
   
 
    # ISDR calculations
    istim = inps2f['istim']
    #istim.flags['C_CONTIGUOUS']
    shsh=istim.shape
    istim=np.ascontiguousarray(istim).ravel()
    inrw = (t - istim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    isdr = inps0f['islev'] * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr=isdr.reshape(shsh)
    isdr = np.sum(isdr, axis=0)
    
    
    istim_di = inps2f['istim_di']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=istim_di.shape
    istim_di=np.ascontiguousarray(istim_di).ravel()
    inrw = (t - istim_di) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    isdr_di = inps0f['islev_di'] * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr_di=isdr_di.reshape(shsh)
    isdr_di = np.sum(isdr_di, axis=0)
    

    # LSDR calculations
    lstim = inps2f['lstim']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=lstim.shape
    lstim=np.ascontiguousarray(lstim).ravel()
    inrw = (t - lstim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    lsdr = inps0f['lslev'] * (
        (1 - dum) * inps1f['sprofil_lr'][ind] + dum * inps1f['sprofil_lr'][ind+1]
    )
    lsdr=lsdr.reshape(shsh)
    lsdr = np.sum(lsdr, axis=0)
    
    # Rates
    
    Tm_di=inps2f['temps_di'][:,flrt] #Tm_di.flags['C_CONTIGUOUS']
    Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
    dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
    ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
    dum=dum-ind
    
    rates=inps2f['rates'][:,ind]*(1-dum)+inps2f['rates'][:,ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
    r_to_di=(Tm_di<inps0f['temp_to_di'])*1
    r_out_di=(Tm_di>inps0f['temp_out_di'])*0.5
    #test=inps1i['l2ind']
    # Reshape y
    y = y.reshape(R, -1).T #y.flags['F_CONTIGUOUS']
    
    # Calculate kk
    
    xx = np.sum(y[inps1i['linds'], :], axis=0)/ Clt 
    xx[Clt == 0] = np.inf   
    kcof = 2.5
    kk = 1 / (1 + np.exp(kcof * (2 * xx - 1)))
    
    # ###########################################################Initialize dydt
    dydt = np.zeros(y.shape,float) #dydt.flags['C_CONTIGUOUS']
    ################################################################################33
 
    
    dydt[inps1i['l1ind'], :]=  - (rates[2, :]+rates[3,:]+lsdr) * y[inps1i['l1ind'], :]  
    dydt[inps1i['l1ind'][1:], :]= dydt[inps1i['l1ind'][1:], :]+ rates[2, :]*y[inps1i['l1ind'][:-1], :] 
    dydt[inps1i['l1ind'][0], :]= dydt[inps1i['l1ind'][0], :]+ (rates[0, :]*kk)*y[inps1i['eind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l2ind'], :]=  - (rates[4, :]+rates[5,:]+lsdr) * y[inps1i['l2ind'], :]  
    dydt[inps1i['l2ind'][1:], :]= dydt[inps1i['l2ind'][1:], :]+ rates[4, :]*y[inps1i['l2ind'][:-1], :] 
    dydt[inps1i['l2ind'][0], :]= dydt[inps1i['l2ind'][0], :]+ rates[2, :]*y[inps1i['l1ind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l3ind'], :]=  - (rates[6, :]+rates[7,:]+lsdr) * y[inps1i['l3ind'], :]  
    dydt[inps1i['l3ind'][1:], :]= dydt[inps1i['l3ind'][1:], :]+ rates[6, :]*y[inps1i['l3ind'][:-1], :] 
    dydt[inps1i['l3ind'][0], :]= dydt[inps1i['l3ind'][0], :]+ rates[4, :]*y[inps1i['l2ind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l4ind'], :]=  - (rates[8, :]+rates[9,:]+lsdr) * y[inps1i['l4ind'], :]  
    dydt[inps1i['l4ind'][1:], :]= dydt[inps1i['l4ind'][1:], :]+ rates[8, :]*y[inps1i['l4ind'][:-1], :] 
    dydt[inps1i['l4ind'][0], :]= dydt[inps1i['l4ind'][0], :]+ rates[6, :]*y[inps1i['l3ind'][-1], :]  
    ################################################################################33
    
    dydt[inps1i['pind'], :]=  - (rates[10, :]+rates[11,:]) * y[inps1i['pind'], :]   
    dydt[inps1i['pind'][1:], :]= dydt[inps1i['pind'][1:], :]+ rates[10, :]*y[inps1i['pind'][:-1], :] 
    dydt[inps1i['pind'][0], :]= dydt[inps1i['pind'][0], :]+ rates[8, :]*y[inps1i['l4ind'][-1], :]  
    
    ################################################################################33
    
    decay=rates[14,:]+rates[12,:]+isdr      
    
    dydt[inps1i['ainds'], :]=  rates[14,:] * y[inps1i['Shainds'],:] - decay * y[inps1i['ainds'],:]
    dydt[inps1i['a1ind'][0], :]=dydt[inps1i['a1ind'][0], :] + 0.5*(1-r_to_di)*rates[10, :]*y[inps1i['pind'][-1],:] 
    dydt[inps1i['a2to5inds'], :]=dydt[inps1i['a2to5inds'], :] + rates[12, :]*y[inps1i['a1to4inds'],:] 
    dydt[inps1i['aindsfirst'], :]=dydt[inps1i['aindsfirst'], :] + r_out_di *y[inps1i['aind_di'],:] 
    
    ################################################################################33
        
    dydt[inps1i['aind_di'], :]=  - (rates[18,:]+r_out_di+isdr_di ) *  y[inps1i['aind_di'], :]   
    dydt[inps1i['aind_di'][1:], :]= dydt[inps1i['aind_di'][1:], :]+ rates[18, :]*y[inps1i['aind_di'][:-1], :] 
    dydt[inps1i['aind_di'][0], :]= dydt[inps1i['aind_di'][0], :]+ 0.5*r_to_di*rates[10, :]*y[inps1i['pind'][-1],:]                                                                         
     
     ################################################################################33
                                                                    
    dydt[inps1i['eind'], :]=  - (rates[0, :]+rates[1,:]) * y[inps1i['eind'], :]  
    dydt[inps1i['eind'][1:], :]= dydt[inps1i['eind'][1:], :]+ rates[0, :]*y[inps1i['eind'][:-1], :] 
    dydt[inps1i['eind'][0], :]= dydt[inps1i['eind'][0], :]+ (rates[14, :]*rates[16, :])* np.sum(y[inps1i['aindslast'], :] , axis=0) 


    # #dydt.flags['C_CONTIGUOUS']
    test=dydt.T.ravel()
    return test



# JIT-compiled function
@njit
def odebackward_PIP_di(t, y, inps0i,inps0f,inps1i,inps1f,inps2i,inps2f):
    
    ff=inps0f['ff']
    tre=ff-t
    flrt=int(np.floor(tre))
    R = int(inps0i['R'])
    
    t0el=inps0f['t0el']
    deltel=inps0f['deltel']
    
    inrw=(tre-t0el)/deltel
    ind=np.maximum(0,int(np.floor(inrw)))
    dum=np.maximum(0, inrw - ind )
    et=(1-dum)*inps2f['en'][ind,:]+dum*inps2f['en'][ind+1,:]
    lt=(1-dum)*inps2f['l'][ind,:]+dum*inps2f['l'][ind+1,:]
   ##############################################
    del_sprofil = inps0f['del_sprofil']

    istim = inps2f['istim']
    #istim.flags['C_CONTIGUOUS']
    shsh=istim.shape
    istim=np.ascontiguousarray(istim).ravel()
    inrw = (tre - istim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    isdr = inps0f['islev'] * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr=isdr.reshape(shsh)
    isdr = np.sum(isdr, axis=0)
    
    
    istim_di = inps2f['istim_di']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=istim_di.shape
    istim_di=np.ascontiguousarray(istim_di).ravel()
    inrw = (tre - istim_di) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    isdr_di = inps0f['islev_di'] * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr_di=isdr_di.reshape(shsh)
    isdr_di = np.sum(isdr_di, axis=0)
    

    # LSDR calculations
    lstim = inps2f['lstim']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=lstim.shape
    lstim=np.ascontiguousarray(lstim).ravel()
    inrw = (tre - lstim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    lsdr = inps0f['lslev'] * (
        (1 - dum) * inps1f['sprofil_lr'][ind] + dum * inps1f['sprofil_lr'][ind+1]
    )
    lsdr=lsdr.reshape(shsh)
    lsdr = np.sum(lsdr, axis=0)
    

   ##############################################
    Kt = inps2f['Kt'][:, flrt]
   
    Cl0t = inps2f['Cl0t']
    if Cl0t.shape[1] > 1:
       Cl0t = Cl0t[:, flrt ]
    else:
       Cl0t = Cl0t[:, 0 ]

    del_delta = inps0f['del_delta']
   
    taucl = inps2f['taucl']
    shsh=taucl.shape
    taucl=np.ascontiguousarray(taucl).ravel()
    inrw = (tre - taucl) / del_delta
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
   
    Clt = inps0i['Nhr'] * inps0f['alpha'] * (
       (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
    )
    Clt=Clt.reshape(shsh)
    Clt = np.sum(Clt, axis=0)
    Clt = np.maximum(Cl0t - Clt, 0) * Kt
   

    #######################################################################
 
    Tm_di=inps2f['temps_di'][:,flrt] #Tm_di.flags['C_CONTIGUOUS']
    Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
    dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
    ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
    dum=dum-ind
    
    rates=inps2f['rates'][:,ind]*(1-dum)+inps2f['rates'][:,ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
    r_to_di=(Tm_di<inps0f['temp_to_di'])*1
    r_out_di=(Tm_di>inps0f['temp_out_di'])*0.5
    
    ###################################################################
       

    dum = (Clt != 0)
    kcof = 2.5
    jdum = 1.0 / (1 + np.exp(2 * kcof * ((lt[dum] / Clt[dum]) - 0.5)))
    neg_djdum_dcldum = 2 * kcof * ((jdum * (1 - jdum)) / Clt[dum])
    rrr = np.zeros(R)
    rrr[dum] = neg_djdum_dcldum
    rr = np.zeros(R)
    rr[dum] = jdum
    
    ##############################################################
    y = y.reshape(R, -1).T #lam.flags['F_CONTIGUOUS']
    dydt = np.zeros(y.shape,float) #dydt.flags['C_CONTIGUOUS']
    ########################################################## 
    rrr=rrr * et * rates[0, :] * y[inps1i['l1ind'][0], :]
    
    dydt[inps1i['l1ind'], :]=  - (rates[2, :]+rates[3,:]+lsdr) * y[inps1i['l1ind'], :]  - rrr 
    dydt[inps1i['l1ind'][:-1], :]= dydt[inps1i['l1ind'][:-1], :]+ rates[2, :]*y[inps1i['l1ind'][1:], :] 
    dydt[inps1i['l1ind'][-1], :]= dydt[inps1i['l1ind'][-1], :]+ rates[2, :]*y[inps1i['l2ind'][0], :] 
    ###########
    
    dydt[inps1i['l2ind'], :]=  - (rates[4, :]+rates[5,:]+lsdr) * y[inps1i['l2ind'], :]   - rrr
    dydt[inps1i['l2ind'][:-1], :]= dydt[inps1i['l2ind'][:-1], :]+ rates[4, :]*y[inps1i['l2ind'][1:], :] 
    dydt[inps1i['l2ind'][-1], :]= dydt[inps1i['l2ind'][-1], :]+ rates[4, :]*y[inps1i['l3ind'][0], :] 
    
    ###########
    
    dydt[inps1i['l3ind'], :]=  - (rates[6, :]+rates[7,:]+lsdr) * y[inps1i['l3ind'], :]  -rrr
    dydt[inps1i['l3ind'][:-1], :]= dydt[inps1i['l3ind'][:-1], :]+ rates[6, :]*y[inps1i['l3ind'][1:], :] 
    dydt[inps1i['l3ind'][-1], :]= dydt[inps1i['l3ind'][-1], :]+ rates[6, :]*y[inps1i['l4ind'][0], :] 
    
    ###########
    
    dydt[inps1i['l4ind'], :]=  - (rates[8, :]+rates[9,:]+lsdr) * y[inps1i['l4ind'], :] - rrr 
    dydt[inps1i['l4ind'][:-1], :]= dydt[inps1i['l4ind'][:-1], :]+ rates[8, :]*y[inps1i['l4ind'][1:], :] 
    dydt[inps1i['l4ind'][-1], :]= dydt[inps1i['l4ind'][-1], :]+ rates[8, :]*y[inps1i['pind'][0], :]  
    ################################################################################33
    
    dydt[inps1i['pind'], :]=  - (rates[10, :]+rates[11,:]) * y[inps1i['pind'], :]   
    dydt[inps1i['pind'][:-1], :]= dydt[inps1i['pind'][:-1], :]+ rates[10, :]*y[inps1i['pind'][1:], :] 
    dydt[inps1i['pind'][-1], :]= dydt[inps1i['pind'][-1], :]+ 0.5*rates[10, :]*(r_to_di*y[inps1i['aind_di'][0],:] + (1-r_to_di)*y[inps1i['a1ind'][0],:])
    
    ################################################################################33
    decay=rates[14,:]+rates[12,:]+isdr      
    
    dydt[inps1i['ainds'], :]=  rates[14,:] * y[inps1i['Shainds_bk'],:] - decay * y[inps1i['ainds'],:]
    dydt[inps1i['a1to4inds'], :]=dydt[inps1i['a1to4inds'], :] + rates[12, :]*y[inps1i['a2to5inds'],:] 
    dydt[inps1i['aindslast'], :]=dydt[inps1i['aindslast'], :] + (rates[14, :]*rates[16, :]) *y[inps1i['eind'][0],:] 
    
    ###################################################################################
    dydt[inps1i['aind_di'], :]=  - (rates[18,:]+r_out_di+isdr_di ) *  y[inps1i['aind_di'], :] + r_out_di *y[inps1i['aindsfirst'],:]  
    dydt[inps1i['aind_di'][:-1], :]= dydt[inps1i['aind_di'][:-1], :]+ rates[18, :]*y[inps1i['aind_di'][1:], :] 
                                                                       
     
     ################################################################################33
                                                                    
    dydt[inps1i['eind'], :]=  - (rates[0, :]+rates[1,:]) * y[inps1i['eind'], :]  
    dydt[inps1i['eind'][:-1], :]= dydt[inps1i['eind'][:-1], :]+ rates[0, :]*y[inps1i['eind'][1:], :] 
    dydt[inps1i['eind'][-1], :]= dydt[inps1i['eind'][-1], :]+ rr * rates[0, :]* y[inps1i['l1ind'][0], :] 

    ###############################################################################
    
    if inps0i['mode']==1:
        
        Wriskova=(1/(1+10*rates[13, :])) * inps1f['weis']*rates[17, :]*(rates[15, :]/rates[13, :])**2
        dydt[inps1i['ainds'], :]=dydt[inps1i['ainds'], :] - Wriskova 
        
    elif inps0i['mode']==2:
        
        dydt[inps1i['ainds'], :]=dydt[inps1i['ainds'], :] - inps1f['weis']
     
    

 # #dydt.flags['C_CONTIGUOUS']
    test=dydt.T.ravel()           
    return test





def dfdci_PIP_di(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f,lam,y,tspan,mxin):    
      
     lam_t = lam[:,::-1]
     
     R=inps0i['R']
     dum=lam.shape[1]
     dfdadrt=np.zeros((R,dum))
     dfdadidrt=np.zeros((R,dum))
     dfdldrt=np.zeros((R,dum))
     laml11=np.zeros((R,dum))
     eend=np.zeros((R,dum))
     L=np.zeros((R,dum))
     
     for rr in range(R):
         ofs=rr*mxin
         dfdadrt[rr,:] = np.sum(lam_t[inps1i['ainds']+ofs,] * y[inps1i['ainds']+ofs,] , axis=0)
         dfdadidrt[rr,:] = np.sum(lam_t[inps1i['aind_di']+ofs,] * y[inps1i['aind_di']+ofs,] , axis=0)
         dfdldrt[rr,:] = np.sum(lam_t[inps1i['linds']+ofs,] * y[inps1i['linds']+ofs,] , axis=0)
         
         laml11[rr,:] = lam_t[inps1i['l1ind'][0]+ofs,]
         eend[rr,:] = y[inps1i['eind'][-1]+ofs,]
         L[rr,:] = np.sum(y[inps1i['linds']+ofs,],axis=0)
     
     
     #### plt.plot(dfdadidrt)
     ########################################################################
     flrt=np.floor(tspan).astype(np.int32)
     Kt = inps2f['Kt'][:, flrt]
    
     Cl0t = inps2f['Cl0t']
     if Cl0t.shape[1] > 1:
        Cl0t = Cl0t[:, flrt ]
     else:
        Cl0t = Cl0t[:, 0:1 ]

     
     del_delta = inps0f['del_delta']   
     ts=tspan[None,:]   
     taucl = inps2f['taucl']
     Clt=0
     for i in range(taucl.shape[0]):
         tuc=taucl[i,:]
         tuc=tuc[:,None]  
         inrw = ((ts-tuc)/del_delta)
         ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
         dum = np.maximum(0, inrw - ind )
         Clt = Clt+inps0i['Nhr'] * inps0f['alpha'] * (
            (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
         )        
     
     Clt = np.maximum(Cl0t - Clt, 0) * Kt
     
     dum = (Clt != 0)
     kcof = 2.5
     jdum = 1.0 / (1 + np.exp(2 * kcof * ((L[dum] / Clt[dum]) - 0.5)))
     djdum_dcldum_e = 2 * kcof * ( jdum * (1 - jdum) * L[dum] * eend[dum] / (Clt[dum]**2) )
     rrr = np.zeros(Clt.shape)
     rrr[dum] = djdum_dcldum_e
     
     
     Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
     dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
     ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
     dum=dum-ind
     gam_el1= inps2f['rates'][0,]
     rates=gam_el1[ind]*(1-dum)+gam_el1[ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
     
     dfdalph_kt=-(rrr * laml11 * rates * Kt)
     
     
     
     return dfdldrt , dfdadrt , dfdadidrt , dfdalph_kt






# JIT-compiled function
@njit
def odeforward_PIP_mh(t, y, inps0i,inps0f,inps1i,inps1f,inps2i,inps2f):
  

    R = int(inps0i['R'])
    flrt=int(np.floor(t))
   
    Kt = inps2f['Kt'][:, flrt]
    
    Cl0t = inps2f['Cl0t']
    if Cl0t.shape[1] > 1:
        Cl0t = Cl0t[:, flrt ]
    else:
        Cl0t = Cl0t[:, 0 ]
        
        

    del_delta = inps0f['del_delta']   
    taucl = inps2f['taucl']
    shsh=taucl.shape
    taucl=np.ascontiguousarray(taucl).ravel()
    inrw = (t - taucl) / del_delta
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )   
    alpha=np.ascontiguousarray(inps2f['alpha']).ravel()
    Clt = inps0i['Nhr'] * alpha * (
        (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
    )
    Clt=Clt.reshape(shsh)
    Clt = np.sum(Clt, axis=0)
    Clt = np.maximum(Cl0t - Clt, 0) * Kt
    
    
  
   
    del_sprofil = inps0f['del_sprofil']
    istim = inps2f['istim']
    #istim.flags['C_CONTIGUOUS']
    shsh=istim.shape
    istim=np.ascontiguousarray(istim).ravel()
    inrw = (t - istim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    islev=np.ascontiguousarray(inps2f['islev']).ravel()
    isdr = islev * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr=isdr.reshape(shsh)
    isdr = np.sum(isdr, axis=0)
    


    lstim = inps2f['lstim']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=lstim.shape
    lstim=np.ascontiguousarray(lstim).ravel()
    inrw = (t - lstim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    lslev=np.ascontiguousarray(inps2f['lslev']).ravel()
    lsdr = lslev * (
        (1 - dum) * inps1f['sprofil_lr'][ind] + dum * inps1f['sprofil_lr'][ind+1]
    )
    lsdr=lsdr.reshape(shsh)
    lsdr = np.sum(lsdr, axis=0)
    
    # Rates
    
   
    Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
    Tm_ma=inps2f['temps_ma'][:,flrt]
    dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
    ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
    dum=dum-ind    
    rates=inps2f['rates'][:,ind]*(1-dum)+inps2f['rates'][:,ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
    RR=round(R/2);       
    r_to_mh= inps0f['fto'] * rates[13,0:RR] *((Tm_ma<inps0f['temp_to_mh_mn'])|(Tm_ma>inps0f['temp_to_mh_mx']));
    r_out_mh= inps0f['fout'] * rates[13,RR:R] *((Tm_ma>inps0f['temp_out_mh_mn']) & (Tm_ma<inps0f['temp_out_mh_mx']));
    
    
    y = y.reshape(R, -1).T #y.flags['F_CONTIGUOUS']
    
 

    lt=np.sum(y[inps1i['linds'], :], axis=0)
    dum = (Clt != 0)
    kcof = 2.5
    kk = np.zeros(R)
    kk[dum] = 1.0 / (1 + np.exp(2 * kcof * ((lt[dum] / Clt[dum]) - 0.5)))
    
    # ###########################################################Initialize dydt
    dydt = np.zeros(y.shape,float) #dydt.flags['C_CONTIGUOUS']
    ################################################################################33
 
    
    dydt[inps1i['l1ind'], :]=  - (rates[2, :]+rates[3,:]+lsdr) * y[inps1i['l1ind'], :]  
    dydt[inps1i['l1ind'][1:], :]= dydt[inps1i['l1ind'][1:], :]+ rates[2, :]*y[inps1i['l1ind'][:-1], :] 
    dydt[inps1i['l1ind'][0], :]= dydt[inps1i['l1ind'][0], :]+ (rates[0, :]*kk)*y[inps1i['eind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l2ind'], :]=  - (rates[4, :]+rates[5,:]+lsdr) * y[inps1i['l2ind'], :]  
    dydt[inps1i['l2ind'][1:], :]= dydt[inps1i['l2ind'][1:], :]+ rates[4, :]*y[inps1i['l2ind'][:-1], :] 
    dydt[inps1i['l2ind'][0], :]= dydt[inps1i['l2ind'][0], :]+ rates[2, :]*y[inps1i['l1ind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l3ind'], :]=  - (rates[6, :]+rates[7,:]+lsdr) * y[inps1i['l3ind'], :]  
    dydt[inps1i['l3ind'][1:], :]= dydt[inps1i['l3ind'][1:], :]+ rates[6, :]*y[inps1i['l3ind'][:-1], :] 
    dydt[inps1i['l3ind'][0], :]= dydt[inps1i['l3ind'][0], :]+ rates[4, :]*y[inps1i['l2ind'][-1], :] 
    
    ###########
    
    dydt[inps1i['l4ind'], :]=  - (rates[8, :]+rates[9,:]+lsdr) * y[inps1i['l4ind'], :]  
    dydt[inps1i['l4ind'][1:], :]= dydt[inps1i['l4ind'][1:], :]+ rates[8, :]*y[inps1i['l4ind'][:-1], :] 
    dydt[inps1i['l4ind'][0], :]= dydt[inps1i['l4ind'][0], :]+ rates[6, :]*y[inps1i['l3ind'][-1], :]  
    ################################################################################33
    
    dydt[inps1i['pind'], :]=  - (rates[10, :]+rates[11,:]) * y[inps1i['pind'], :]   
    dydt[inps1i['pind'][1:], :]= dydt[inps1i['pind'][1:], :]+ rates[10, :]*y[inps1i['pind'][:-1], :] 
    dydt[inps1i['pind'][0], :]= dydt[inps1i['pind'][0], :]+ rates[8, :]*y[inps1i['l4ind'][-1], :]  
    
    ################################################################################33
    
    decay=rates[14,:]+rates[12,:]+isdr +  np.hstack((r_to_mh, r_out_mh))    
    
    dydt[inps1i['ainds'], :]=  rates[14,:] * y[inps1i['Shainds'],:] - decay * y[inps1i['ainds'],:]
    dydt[inps1i['a1ind'][0], :]=dydt[inps1i['a1ind'][0], :] + 0.5 * rates[10, :]*y[inps1i['pind'][-1],:] 
    dydt[inps1i['a2to5inds'], :]=dydt[inps1i['a2to5inds'], :] + rates[12, :]*y[inps1i['a1to4inds'],:] 
   
   
    dydt[inps1i['ainds'], :]=dydt[inps1i['ainds'], :] + np.hstack(( r_out_mh * y[inps1i['ainds'], RR:R] , r_to_mh * y[inps1i['ainds'], 0:RR] ))
    
    ################################################################################33
                                                                           

                                                                    
    dydt[inps1i['eind'], :]=  - (rates[0, :]+rates[1,:]) * y[inps1i['eind'], :]  
    dydt[inps1i['eind'][1:], :]= dydt[inps1i['eind'][1:], :]+ rates[0, :]*y[inps1i['eind'][:-1], :] 
    dydt[inps1i['eind'][0], :]= dydt[inps1i['eind'][0], :]+ (rates[14, :]*rates[16, :])* np.sum(y[inps1i['aindslast'], :] , axis=0) 


    # #dydt.flags['C_CONTIGUOUS']
    test=dydt.T.ravel()
    return test







# JIT-compiled function
@njit
def odebackward_PIP_mh(t, y, inps0i,inps0f,inps1i,inps1f,inps2i,inps2f):
    
    ff=inps0f['ff']
    tre=ff-t
    flrt=int(np.floor(tre))
    R = int(inps0i['R'])
    
    t0el=inps0f['t0el']
    deltel=inps0f['deltel']
    
    inrw=(tre-t0el)/deltel
    ind=np.maximum(0,int(np.floor(inrw)))
    dum=np.maximum(0, inrw - ind )
    et=(1-dum)*inps2f['en'][ind,:]+dum*inps2f['en'][ind+1,:]
    lt=(1-dum)*inps2f['l'][ind,:]+dum*inps2f['l'][ind+1,:]
   ##############################################
    del_sprofil = inps0f['del_sprofil']

    istim = inps2f['istim']
    #istim.flags['C_CONTIGUOUS']
    shsh=istim.shape
    istim=np.ascontiguousarray(istim).ravel()
    inrw = (tre - istim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    islev=np.ascontiguousarray(inps2f['islev']).ravel()
    isdr = islev * (
        (1 - dum) * inps1f['sprofil_ins'][ind] + dum * inps1f['sprofil_ins'][ind+1]
    )
    isdr=isdr.reshape(shsh)
    isdr = np.sum(isdr, axis=0)
    

    
    

    # LSDR calculations
    lstim = inps2f['lstim']
    #istim_di.flags['C_CONTIGUOUS']
    shsh=lstim.shape
    lstim=np.ascontiguousarray(lstim).ravel()
    inrw = (tre - lstim) / del_sprofil
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    lslev=np.ascontiguousarray(inps2f['lslev']).ravel()
    lsdr = lslev * (
        (1 - dum) * inps1f['sprofil_lr'][ind] + dum * inps1f['sprofil_lr'][ind+1]
    )
    lsdr=lsdr.reshape(shsh)
    lsdr = np.sum(lsdr, axis=0)
    

   ##############################################
    Kt = inps2f['Kt'][:, flrt]
   
    Cl0t = inps2f['Cl0t']
    if Cl0t.shape[1] > 1:
       Cl0t = Cl0t[:, flrt ]
    else:
       Cl0t = Cl0t[:, 0 ]

    del_delta = inps0f['del_delta']
   
    taucl = inps2f['taucl']
    shsh=taucl.shape
    taucl=np.ascontiguousarray(taucl).ravel()
    inrw = (tre - taucl) / del_delta
    ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
    dum = np.maximum(0, inrw - ind )
    alpha=np.ascontiguousarray(inps2f['alpha']).ravel()
    Clt = inps0i['Nhr'] * alpha * (
       (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
    )
    Clt=Clt.reshape(shsh)
    Clt = np.sum(Clt, axis=0)
    Clt = np.maximum(Cl0t - Clt, 0) * Kt
   

    #######################################################################
    Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
    Tm_ma=inps2f['temps_ma'][:,flrt]
    dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
    ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
    dum=dum-ind    
    rates=inps2f['rates'][:,ind]*(1-dum)+inps2f['rates'][:,ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
    RR=round(R/2)      
    r_to_mh= inps0f['fto'] * rates[13,0:RR] *((Tm_ma<inps0f['temp_to_mh_mn'])|(Tm_ma>inps0f['temp_to_mh_mx']));
    r_out_mh= inps0f['fout'] * rates[13,RR:R] *((Tm_ma>inps0f['temp_out_mh_mn']) & (Tm_ma<inps0f['temp_out_mh_mx']));
        
    ###################################################################
       

    dum = (Clt != 0)
    kcof = 2.5
    jdum = 1.0 / (1 + np.exp(2 * kcof * ((lt[dum] / Clt[dum]) - 0.5)))
    neg_djdum_dcldum = 2 * kcof * ((jdum * (1 - jdum)) / Clt[dum])
    rrr = np.zeros(R)
    rrr[dum] = neg_djdum_dcldum
    rr = np.zeros(R)
    rr[dum] = jdum
    
 
    ##############################################################
    y = y.reshape(R, -1).T #lam.flags['F_CONTIGUOUS']
    dydt = np.zeros(y.shape,float) #dydt.flags['C_CONTIGUOUS']
    ########################################################## 
    rrr=rrr * et * rates[0, :] * y[inps1i['l1ind'][0], :]
    
    dydt[inps1i['l1ind'], :]=  - (rates[2, :]+rates[3,:]+lsdr) * y[inps1i['l1ind'], :]  - rrr 
    dydt[inps1i['l1ind'][:-1], :]= dydt[inps1i['l1ind'][:-1], :]+ rates[2, :]*y[inps1i['l1ind'][1:], :] 
    dydt[inps1i['l1ind'][-1], :]= dydt[inps1i['l1ind'][-1], :]+ rates[2, :]*y[inps1i['l2ind'][0], :] 
    ###########
    
    dydt[inps1i['l2ind'], :]=  - (rates[4, :]+rates[5,:]+lsdr) * y[inps1i['l2ind'], :]   - rrr
    dydt[inps1i['l2ind'][:-1], :]= dydt[inps1i['l2ind'][:-1], :]+ rates[4, :]*y[inps1i['l2ind'][1:], :] 
    dydt[inps1i['l2ind'][-1], :]= dydt[inps1i['l2ind'][-1], :]+ rates[4, :]*y[inps1i['l3ind'][0], :] 
    
    ###########
    
    dydt[inps1i['l3ind'], :]=  - (rates[6, :]+rates[7,:]+lsdr) * y[inps1i['l3ind'], :]  -rrr
    dydt[inps1i['l3ind'][:-1], :]= dydt[inps1i['l3ind'][:-1], :]+ rates[6, :]*y[inps1i['l3ind'][1:], :] 
    dydt[inps1i['l3ind'][-1], :]= dydt[inps1i['l3ind'][-1], :]+ rates[6, :]*y[inps1i['l4ind'][0], :] 
    
    ###########
    
    dydt[inps1i['l4ind'], :]=  - (rates[8, :]+rates[9,:]+lsdr) * y[inps1i['l4ind'], :] - rrr 
    dydt[inps1i['l4ind'][:-1], :]= dydt[inps1i['l4ind'][:-1], :]+ rates[8, :]*y[inps1i['l4ind'][1:], :] 
    dydt[inps1i['l4ind'][-1], :]= dydt[inps1i['l4ind'][-1], :]+ rates[8, :]*y[inps1i['pind'][0], :]  
    ################################################################################33
    
    dydt[inps1i['pind'], :]=  - (rates[10, :]+rates[11,:]) * y[inps1i['pind'], :]   
    dydt[inps1i['pind'][:-1], :]= dydt[inps1i['pind'][:-1], :]+ rates[10, :]*y[inps1i['pind'][1:], :] 
    dydt[inps1i['pind'][-1], :]= dydt[inps1i['pind'][-1], :]+ 0.5*rates[10, :] * y[inps1i['a1ind'][0],:]
    
    ################################################################################33
    decay=rates[14,:]+rates[12,:]+isdr + np.hstack((r_to_mh, r_out_mh))     
    
    dydt[inps1i['ainds'], :]=  rates[14,:] * y[inps1i['Shainds_bk'],:] - decay * y[inps1i['ainds'],:] + \
    np.hstack(( r_to_mh * y[inps1i['ainds'], RR:R] , r_out_mh * y[inps1i['ainds'], 0:RR] ))
    
    dydt[inps1i['a1to4inds'], :]=dydt[inps1i['a1to4inds'], :] + rates[12, :]*y[inps1i['a2to5inds'],:] 

    
    dydt[inps1i['aindslast'], :]=dydt[inps1i['aindslast'], :] + (rates[14, :]*rates[16, :]) *y[inps1i['eind'][0],:] 
    
    ###################################################################################

                                                                       
     
     ################################################################################33
                                                                    
    dydt[inps1i['eind'], :]=  - (rates[0, :]+rates[1,:]) * y[inps1i['eind'], :]  
    dydt[inps1i['eind'][:-1], :]= dydt[inps1i['eind'][:-1], :]+ rates[0, :]*y[inps1i['eind'][1:], :] 
    dydt[inps1i['eind'][-1], :]= dydt[inps1i['eind'][-1], :]+ rr * rates[0, :]* y[inps1i['l1ind'][0], :] 

    ###############################################################################
    
    if inps0i['mode']==1:
        
        Wriskova=(1/(1+10*rates[13, :])) * inps1f['weis']*rates[17, :]*(rates[15, :]/rates[13, :])**2
        dydt[inps1i['ainds'], :]=dydt[inps1i['ainds'], :] - Wriskova 
        
    elif inps0i['mode']==2:
        
        dydt[inps1i['ainds'], :]=dydt[inps1i['ainds'], :] - inps1f['weis']
     
    

 # #dydt.flags['C_CONTIGUOUS']
    test=dydt.T.ravel()           
    return test






def dfdci_PIP_mh(inps0i,inps0f,inps1i,inps1f,inps2i,inps2f,lam,y,tspan,mxin):    
      
     lam_t = lam[:,::-1]
     
     R=inps0i['R']
     dum=lam.shape[1]
     dfdadrt=np.zeros((R,dum))
     dfdldrt=np.zeros((R,dum))
     laml11=np.zeros((R,dum))
     eend=np.zeros((R,dum))
     L=np.zeros((R,dum))
     
     for rr in range(R):
         ofs=rr*mxin
         dfdadrt[rr,:] = np.sum(lam_t[inps1i['ainds']+ofs,] * y[inps1i['ainds']+ofs,] , axis=0)
         dfdldrt[rr,:] = np.sum(lam_t[inps1i['linds']+ofs,] * y[inps1i['linds']+ofs,] , axis=0)
         
         laml11[rr,:] = lam_t[inps1i['l1ind'][0]+ofs,]
         eend[rr,:] = y[inps1i['eind'][-1]+ofs,]
         L[rr,:] = np.sum(y[inps1i['linds']+ofs,],axis=0)
     
     
     #### plt.plot(dfdadidrt)
     ########################################################################
     flrt=np.floor(tspan).astype(np.int32)
     Kt = inps2f['Kt'][:, flrt]
    
     Cl0t = inps2f['Cl0t']
     if Cl0t.shape[1] > 1:
        Cl0t = Cl0t[:, flrt ]
     else:
        Cl0t = Cl0t[:, 0:1 ]

     
     del_delta = inps0f['del_delta']   
     ts=tspan[None,:]   
     taucl = inps2f['taucl']
     Clt=0
     for i in range(taucl.shape[0]):
         tuc=taucl[i,:]
         alpha=inps2f['alpha'][i,] 
         alpha=alpha[:,None]
         tuc=tuc[:,None]  
         inrw = ((ts-tuc)/del_delta)
         ind =  np.maximum(0, np.floor(inrw).astype(np.int32))
         dum = np.maximum(0, inrw - ind )
         Clt = Clt+inps0i['Nhr'] * alpha * (
            (1 - dum) * inps1f['deltaprofil'][ind] + dum * inps1f['deltaprofil'][ind+1]
         )        
     
     Clt = np.maximum(Cl0t - Clt, 0) * Kt
     
     dum = (Clt != 0)
     kcof = 2.5
     jdum = 1.0 / (1 + np.exp(2 * kcof * ((L[dum] / Clt[dum]) - 0.5)))
     djdum_dcldum_e = 2 * kcof * ( jdum * (1 - jdum) * L[dum] * eend[dum] / (Clt[dum]**2) )
     rrr = np.zeros(Clt.shape)
     rrr[dum] = djdum_dcldum_e
     
     
     Tm=inps2f['temps'][:,flrt] #Tm.flags['C_CONTIGUOUS']
     dum=((Tm+70)/0.5) #dum.flags['C_CONTIGUOUS']
     ind=np.floor(dum).astype(np.int32) #ind.flags['C_CONTIGUOUS']
     dum=dum-ind
     gam_el1= inps2f['rates'][0,]
     rates=gam_el1[ind]*(1-dum)+gam_el1[ind+1]*(dum)##rates.flags['F_CONTIGUOUS']
     
     dfdalph_kt=-(rrr * laml11 * rates * Kt)
     
     
     
     return dfdldrt , dfdadrt  , dfdalph_kt










