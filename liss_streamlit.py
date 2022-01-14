import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def tvf(T,A,B,C):
    return A + B/(T-C)

def tg_from_tvf(A,B,C):
    return B/(12-A)+C

st.set_page_config(layout="wide")

st.title('Liss: Fit of viscosity data with VFT or Adam-Gibbs equation')

with st.sidebar.form(key='my_form'):
    uploaded_file = st.file_uploader("Choose a two columns CSV file with headers T_K and viscosity")

    option = st.selectbox(
         'Which equation you want to use?',
         ('VFT', 'ADAM-GIBBS'))

    option2 = st.selectbox(
         'Which regression method?',
         ('trf', 'dogbox', 'lm'))

    st.write('For Adam-Gibbs calculation, please enter the parameters for CpConf:')
    ap_ = st.number_input('ap:')
    b_ = st.number_input('b:')
    #Tg = st.number_input('Tg:')

    st.form_submit_button()

if uploaded_file is not None:
     # To read file as bytes:
     data = pd.read_csv(uploaded_file)

     # fit TVF
     popt, pcov = curve_fit(tvf, data.loc[:,"T_K"], data.loc[:,"viscosity"], p0 = [-4.5, 8000, 500], method=option2, bounds = ([-20,0,0],[20, np.inf, np.inf]))
     perr = np.sqrt(np.diag(pcov))

     # RMSE calculation
     y_calc = tvf(data.loc[:,"T_K"], *popt)
     RMSE = np.sqrt(np.mean((data.loc[:,"viscosity"].ravel()-y_calc.ravel())**2))

     # get TG => NOPE, get it from user.
     Tg = tg_from_tvf(*popt)

     if option == 'ADAM-GIBBS':
         # fit AG
         ag = lambda T,Ae,Be,ScTg: Ae + Be/(T*(ScTg + ap_*(np.log(T)-np.log(Tg)) + b_*(T-Tg)))
         popt_ag, pcov_ag = curve_fit(ag, data.loc[:,"T_K"], data.loc[:,"viscosity"], p0 = [-3.5, 80000, 5.0], method=option2, bounds = ([-20,0,0],[20, np.inf, np.inf]))
         perr_ag = np.sqrt(np.diag(pcov_ag))

         y_calc = ag(data.loc[:,"T_K"], *popt_ag)
         RMSE = np.sqrt(np.mean((data.loc[:,"viscosity"].ravel()-y_calc.ravel())**2))

     # for the plot: a nice X axis interpolating values
     x_fit = np.arange(np.min(data.loc[:,"T_K"]), np.max(data.loc[:,"T_K"]))

     st.subheader("Results")

     if option == "VFT":
         st.write("Equation VFT selected, calculated parameters are:")
         col1, col2, col3, col4 = st.columns(4)
         with col1:
             st.metric('A', "{:.2f} +/- {:.1f}".format(popt[0], perr[0]))
         with col2:
             st.metric('B', "{:.1f} +/- {:.1f}".format(popt[1], perr[1]))
         with col3:
             st.metric('C', "{:.1f} +/- {:.1f}".format(popt[2], perr[2]))
         with col4:
             st.metric('RMSE', '{:.2f}'.format(RMSE))
     elif option == "ADAM-GIBBS":
         st.write("Equation Adam-Gibbs selected, calculated parameters are:")
         col1, col2, col3, col4 = st.columns(4)
         with col1:
             st.metric('Ae', "{:.2f} +/- {:.2f}".format(popt_ag[0],perr_ag[0]))
         with col2:
             st.metric('Be', "{:.1f} +/- {:.1f}".format(popt_ag[1], perr_ag[1]))
         with col3:
             st.metric('Sconf(Tg)', "{:.1f} +/- {:.1f}".format(popt_ag[2], perr_ag[2]))
         with col4:
             st.metric('RMSE', '{:.2f}'.format(RMSE))

     fig = make_subplots(rows=1, cols=2,horizontal_spacing = 0.1,subplot_titles=("Melt viscosity", "Residuals"))
     fig.add_trace(
         go.Scatter(mode='markers',x=10000/data.loc[:,"T_K"], y=data.loc[:,"viscosity"],name="data", legendgroup=1), row=1, col=1)

     if option == "VFT":
        fig.add_trace(go.Scatter(x=10000/x_fit, y=tvf(x_fit, *popt),name="TVF fit", legendgroup=1), row=1, col=1)
        fig.add_trace(go.Scatter(mode='markers',x=10000/data.loc[:,"T_K"], y=tvf(data.loc[:,"T_K"], *popt)-data.loc[:,"viscosity"],name="residuals", legendgroup=2), row=1, col=2)
     if option == "ADAM-GIBBS":
        fig.add_trace(go.Scatter(x=10000/x_fit, y=ag(x_fit, *popt_ag),name="AG fit", legendgroup=1), row=1, col=1)
        fig.add_trace(go.Scatter(mode='markers',x=10000/data.loc[:,"T_K"], y=ag(data.loc[:,"T_K"], *popt_ag)-data.loc[:,"viscosity"],name="residuals", legendgroup=2), row=1, col=2)

     # Update xaxis properties
     fig.update_xaxes(title_text=r'10000/T, K', row=1, col=1)
     fig.update_yaxes(title_text=r"log<sub>10</sub> viscosity", row=1, col=1)

     # Update xaxis properties
     fig.update_xaxes(title_text=r'10000/T, K', row=1, col=2)
     fig.update_yaxes(title_text=r"residuals, log10 units", row=1, col=2)

     fig.update_layout(autosize=True)

     st.plotly_chart(fig)
     ###
     # Print residuals
     ###

     data["Calculated"] = y_calc
     data["RMSE"] = np.sqrt((data.loc[:,"viscosity"].ravel()-y_calc.ravel())**2)

     with st.expander("Table:"):
         st.table(data.round(decimals=2))
