# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:41:23 2015

@author: Charles LE LOSQ
Equations for viscosity and heat capacity
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def heatcp(chimie,Tg = None,Cp_glass = "R1987"):
    """ Return the parameters a and b for the melt configurational heat capacity
    Parameters
    ==========
    datachem : Pandas Dataframe of Length N
        Chemical composition of the melts in mol%
    Tg : ndarray of length N
        Glass transition T in K
        
    Option
    ======
    Cp_glass : string
        Choose either "R1987" for the model of Richet 1987 or "3R" for the Dulong-Petit limit (no Tg needed in this case)

    Returns
    =======
    ap : ndarray

    b : ndarray

    """
    #Chimie du verre étudié
    SiO2 = np.asarray(chimie["sio2"])
    Al2O3 = np.asarray(chimie["al2o3"])
    Na2O = np.asarray(chimie["na2o"])
    K2O = np.asarray(chimie["k2o"])
    MgO = np.asarray(chimie["mgo"])
    CaO = np.asarray(chimie["cao"])
    
    if Cp_glass == "R1987":

        if Tg is None:
            raise ValueError("You should provided the Tg with the R1987 model")
        else:
            #Calcul des coeffs du Cp; avec les valeurs de Richet, CG 62, 1987, 111-124 et Richet, GCA 49, 1985, 471
            Cpg = SiO2*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);
    
    elif Cp_glass == "3R":
        at_gfu = 3*SiO2 + 5*Al2O3 + 3*Na2O + 3*K2O + 2*MgO + 2*MgO
        Cpg = 3*8.314*at_gfu
    else:
         raise ValueError("Choose between R1987 or 3R")
    
    aCpl = 81.37*SiO2 + 27.21*Al2O3 + 100.6*Na2O + 50.13*K2O + SiO2*(K2O*K2O)*151.7 + 86.05*CaO + 85.78*MgO
    bCpl = 0.0943*Al2O3 + 0.01578*K2O
    ap = aCpl - Cpg
    b = bCpl

    return ap, b

def heatcp_AbOr(datachem,Tg):
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]

    XK = K2O/(K2O+Na2O)

    #Calcul des coeffs du Cp; avec les valeurs de Richet and Bottinga 1984 for Albite Orthoclase melts; for glasses Cp model of Richet (1987)
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);

    #Cpg = 3*8.314532 * (0.25+0.25+0.75+2)
    aCpl = 75.168*(1-XK) + 65.46*XK
    bCpl = 0.0107*(1-XK) + XK*0.0155
    ap = aCpl - Cpg
    b = bCpl

    return ap, b

def heatcp_NephKals(datachem,Tg):
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]

    XK = K2O/(K2O+Na2O)

    #Calcul des coeffs du Cp; avec les valeurs de Richet and Bottinga 1984 for Albite Orthoclase melts; for glasses Cp model of Richet (1987)
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);

    aCpl = 86.1#*(1-XK) + 67.3875*XK
    bCpl = 0.0130#*(1-XK) + XK*0.0188
    ap = aCpl - Cpg
    b = bCpl

    return ap, b


def heatcp_wo1991(datachem,Tg):
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]

    XK = CaO/(CaO+MgO)

    #Calcul des coeffs du Cp; avec les valeurs de Richet and Bottinga 1984 for Albite Orthoclase melts; for glasses Cp model of Richet (1987)
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);

    aCpl = (167.15*(1-XK) + 167.43*XK)/3*2
    bCpl = 0
    ap = aCpl - Cpg;
    b = bCpl;

    return ap, b

def heatcp_ks(datachem,Tg): # For using the recommendation of Richet, 1985 for K2O-SiO2 melts
    #Chimie du verre étudié
    SiO2 = datachem[0,0]
    Al2O3 = datachem[1,0]
    Na2O = datachem[2,0]
    K2O = datachem[3,0]
    MgO = datachem[4,0]
    CaO = datachem[5,0]

    #Calcul des coeffs du Cp; avec les valeurs de Richet, CG 62, 1987, 111-124 et Richet, GCA 49, 1985, 471
    Cpg = SiO2/100.0*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3/100.0* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O/100.0*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O/100.0*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO/100.0*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO/100*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);

    xsi = SiO2/100
    xk = K2O/100

    Wsk = 151.7 #J/ K mol

    aCpl = xsi * 81.37 + xk * 50.13 + Wsk*xsi*xk*xk
    bCpl = xk * 0.01578

    ap = aCpl - Cpg
    b = bCpl

    return ap, b

def heatcp_mcmc(SiO2, Al2O3, Na2O, K2O, MgO, CaO, Tg = None, Cp_glass = "R1987"):
    """ Return the parameters a and b for the melt configurational heat capacity
    Parameters
    ==========
    datachem : Pandas Dataframe of Length N
        Chemical composition of the melts in mol%
    Tg : ndarray of length N
        Glass transition T in K
        
    Option
    ======
    Cp_glass : string
        Choose either "R1987" for the model of Richet 1987 or "3R" for the Dulong-Petit limit (no Tg needed in this case)

    Returns
    =======
    ap : ndarray

    b : ndarray

    """

    if Cp_glass == "R1987":

        if Tg is None:
            raise ValueError("You should provided the Tg with the R1987 model")
        else:
            #Calcul des coeffs du Cp; avec les valeurs de Richet, CG 62, 1987, 111-124 et Richet, GCA 49, 1985, 471
            Cpg = SiO2*(127.2 - 0.010777*Tg + 431270.0/Tg**2 -1463.9/Tg**0.5) + Al2O3* (175.491 -0.005839*Tg -1347000./Tg**2 -1370.0/Tg**0.5) + K2O*(84.323 +0.000731*Tg -829800.0/Tg**2) + Na2O*(70.884 +0.02611*Tg -358200.0/Tg**2) +CaO*(39.159 + 0.018650*Tg -152300.0/Tg**2) + MgO*(46.704 + 0.011220*Tg - 1328000.0/Tg**2);
    
    elif Cp_glass == "3R":
        at_gfu = 3*SiO2 + 5*Al2O3 + 3*Na2O + 3*K2O + 2*MgO + 2*MgO
        Cpg = 3*8.314*at_gfu
    else:
         raise ValueError("Choose between R1987 or 3R")
    
    aCpl = 81.37*SiO2 + 27.21*Al2O3 + 100.6*Na2O + 50.13*K2O + SiO2*(K2O*K2O)*151.7 + 86.05*CaO + 85.78*MgO
    bCpl = 0.0943*Al2O3 + 0.01578*K2O
    ap = aCpl - Cpg
    b = bCpl

    return ap, b
