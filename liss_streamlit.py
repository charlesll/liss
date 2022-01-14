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
    st.form_submit_button()

col1, col2 = st.columns(2)
with col1:
    option = st.selectbox(
         'Which equation you want to use?',
         ('VFT', 'ADAM-GIBBS'))

if option == 'ADAM-GIBBS':
    with col2:
        ap = st.number_input('Enter ap to calculate Cpconf')
        b = st.number_input('Enter b to calculate Cp conf')

if uploaded_file is not None:
     # To read file as bytes:
     data = pd.read_csv(uploaded_file)

     # fit TVF
     popt, pcov = curve_fit(tvf, data.loc[:,"T_K"], data.loc[:,"viscosity"], p0 = [-4.5, 8000, 500], method="dogbox", bounds = ([-20,0,0],[20, np.inf, np.inf]))
     perr = np.sqrt(np.diag(pcov))

     # RMSE calculation
     y_calc = tvf(data.loc[:,"T_K"], *popt)
     RMSE = np.sqrt(np.mean((data.loc[:,"viscosity"].ravel()-y_calc.ravel())**2))

     # get TG
     Tg = tg_from_tvf(*popt)

     if option == 'ADAM-GIBBS':
         # fit AG
         ag = lambda T,A,B,ScTg: A + B/(ScTg + ap*np.log(T/Tg) + b*(T-Tg))
         popt_ag, pcov_ag = curve_fit(ag, data.loc[:,"T_K"], data.loc[:,"viscosity"], p0 = [-3.5, 8000, 10.0], bounds = ([-20,0,0],[20, np.inf, np.inf]))
         perr_ag = np.sqrt(np.diag(pcov_ag))

         y_calc_ag = ag(data.loc[:,"T_K"], *popt_ag)
         RMSE_ag = np.sqrt(np.mean((data.loc[:,"viscosity"].ravel()-y_calc_ag.ravel())**2))

     # for the plot: a nice X axis interpolating values
     x_fit = np.arange(np.min(data.loc[:,"T_K"]), np.max(data.loc[:,"T_K"]))

     fig = go.Figure()
     fig.add_trace(
         go.Scatter(mode='markers',x=10000/data.loc[:,"T_K"], y=data.loc[:,"viscosity"],name="data", legendgroup=1))

     if option == "VFT":
         fig.add_trace(
             go.Scatter(x=10000/x_fit, y=tvf(x_fit, *popt),name="TVF", legendgroup=1))
     elif option == "ADAM-GIBBS":
         fig.add_trace(
            go.Scatter(x=10000/x_fit, y=ag(x_fit, *popt_ag),name="AG", legendgroup=1))

     fig.update_layout(
        title="Viscosity data fit",
        xaxis_title="10000/T",
        yaxis_title="log10 Viscosity",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
     st.plotly_chart(fig)

     ###
     # Print residuals
     ###
     if option == "VFT":
         data["TVF_calc"] = y_calc
         data["TVF_RMSE"] = np.sqrt((data.loc[:,"viscosity"].ravel()-y_calc.ravel())**2)

     elif option == "ADAM-GIBBS":
         data["AG_calc"] = y_calc_ag
         data["AG_RMSE"] = np.sqrt((data.loc[:,"viscosity"].ravel()-y_calc_ag.ravel())**2)

     st.table(data.round(decimals=2))
