import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys, getopt
import os

from configparser import ConfigParser

def tvf(T,A,B,C):
    return A + B/(T-C)

def tg_from_tvf(A,B,C):
    return B/(12-A)+C

def main(inputfile, figurefile, outputfile, ap, b):

    # we test if the folder "in" exists
    if not os.path.exists(inputfile):
        raise ValueError('The data file does not exist.')
        sys.exit(1)

    print("###########################")
    print('Data file is ', inputfile)
    print('Figure file is ', figurefile)
    print('Output file is ', outputfile)
    print("###########################")
    print('')

    # import data
    data = pd.read_csv(inputfile)

    # fit TVF
    popt, pcov = curve_fit(tvf, data.loc[:,"T_K"], data.loc[:,"n_pas"], p0 = [-4.5, 8000, 500], method="dogbox", bounds = ([-20,0,0],[20, np.inf, np.inf]))
    perr = np.sqrt(np.diag(pcov))

    # RMSE calculation
    y_calc = tvf(data.loc[:,"T_K"], *popt)
    RMSE = np.sqrt(np.mean((data.loc[:,"n_pas"].ravel()-y_calc.ravel())**2))

    # get TG
    Tg = tg_from_tvf(*popt)

    # fit AG
    ag = lambda T,A,B,ScTg: A + B/(ScTg + ap*np.log(T/Tg) + b*(T-Tg))
    popt_ag, pcov_ag = curve_fit(ag, data.loc[:,"T_K"], data.loc[:,"n_pas"], p0 = [-3.5, 8000, 10.0], bounds = ([-20,0,0],[20, np.inf, np.inf]))
    perr_ag = np.sqrt(np.diag(pcov_ag))

    y_calc_ag = ag(data.loc[:,"T_K"], *popt_ag)
    RMSE_ag = np.sqrt(np.mean((data.loc[:,"n_pas"].ravel()-y_calc_ag.ravel())**2))

    # for the plot: a nice X axis interpolating values
    x_fit = np.arange(np.min(data.loc[:,"T_K"]), np.max(data.loc[:,"T_K"]))

    plt.figure(figsize = (5,5))
    plt.plot(10000/data.loc[:,"T_K"], data.loc[:,"n_pas"], "sk", mfc="none")
    plt.plot(10000/x_fit, tvf(x_fit, *popt), "b--")
    plt.plot(10000/x_fit, ag(x_fit, *popt_ag), "g:")
    plt.annotate("A: {:.2f} +/- {:.2f} \nB: {:.1f} +/- {:.1f}\nC: {:.1f} +/- {:.1f} \nRMSE: {:.2f}".format(popt[0], perr[0], popt[1], perr[1], popt[2], perr[2], RMSE),
                 xy=(0.05,0.8),
                 xycoords="axes fraction", fontsize=12, color="blue")
    plt.annotate("\nAe: {:.2f} +/- {:.2f}\nBe: {:.1f} +/- {:.1f}\nScTg: {:.1f} +/- {:.1f}\nRMSE: {:.2f}".format(popt_ag[0], perr_ag[0], popt_ag[1], perr_ag[1], popt_ag[2], perr_ag[2], RMSE_ag),
                 xy=(0.05,0.6),
                 xycoords="axes fraction", fontsize=12, color="green")
    plt.xlabel("10$^4$/T, K$^{-1}$")
    plt.ylabel("log$_{10}$ viscosity")
    plt.tight_layout()
    plt.savefig(figurefile)
    plt.show()

    data["TVF_calc"] = y_calc
    data["TVF_RMSE"] = np.sqrt((data.loc[:,"n_pas"].ravel()-y_calc.ravel())**2)

    data["AG_calc"] = y_calc_ag
    data["AG_RMSE"] = np.sqrt((data.loc[:,"n_pas"].ravel()-y_calc_ag.ravel())**2)

    data.round(decimals=2).to_csv(outputfile)

if __name__ == "__main__":
    #Read config.ini file
    config_object = ConfigParser()
    config_object.read("configuration.ini")

    #Get inputfile
    paths = config_object["CHEMINS"]
    inputfile = paths["fichier_donnees"]
    figurefile = paths["sauvegarde_figure"]
    outputfile =  paths["sauvegarde_resultats"]

    AG_CP = config_object["Cp_pour_AdamGibbs"]
    ap = float(AG_CP["ap"])
    b = float(AG_CP["b"])

    main(paths["fichier_donnees"],
         paths["sauvegarde_figure"],
         paths["sauvegarde_resultats"], ap, b)
