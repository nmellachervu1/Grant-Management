#!/usr/bin/env python
# coding: utf-8

# In[5]:




from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import io
import base64
import math


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#setting up CORS
from flask_cors import CORS


#Langchain AI
from dotenv import load_dotenv
import os
import getpass


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


class Document:
   def __init__(self, content):
       self.page_content = content
       self.metadata = {}


def generate_torch_kwargs():
   # run torch models on CPU, and disable progress bars for all model stages except training.
   return {
       "pl_trainer_kwargs": {
           "accelerator": "cpu",
           "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
       }
   }


from darts.models import (
   VARIMA,
   BlockRNNModel,
   NBEATSModel,
   RNNModel,
   XGBModel
)


from darts.utils.callbacks import TFMProgressBar


# Set the matplotlib backend to 'Agg'
plt.switch_backend('Agg')


app = Flask(__name__)
CORS(app)
# List of grants
grants = [
   '20NU2GGH0023029390FKV2021', '22R43GH0023699390JGK2022', '22R43GH0023899390JLL2022', '22R43GH002391CV9390JLL2022',
   '22R43GH002392CV9390JLL2022', '21NU2GGH0023399390FKN2021', '21NU2GGH0023399390HCW2021',
   '21R43GH00236793901A32021', '21R43GH002367939ZZMF2021', '21R43GH00236893901KB2021',
   '21R43GH002368939ZZMF2021', '19NU2GGH00219423HOP9390HEC2022', '21NU2GGH0023789390K872023', '20NU2HGH000006C39390GAM2021', '20NU2GGH00231221C39390G802021'
]


SA_grants = [
   '20NU2GGH0023029390FKV2021', '21NU2GGH00235022PEC69390J6V2022', '19NU2GGH00219423HOP9390HEC2022',
   '19NU2GGH00219419HQTB9390FKV2020', '21NU2GGH0023789390K872023', '22NU2GGH0024369390HD32022'
]


India_grants = [
   '20NU2HGH000006C39390GAM2021', '20NU2GGH00231221C39390G802021', '21NU2HGH000088C39390GDP2021',
   '20NU2HGH0000049390EQL2020', '20NU2HGH00000693906BX2021'
]


Ethiopia_grants = [
   '20NU2HGH000077C39390GFF2021', '20NU2HGH000072CV9390ETK2020', '20NU2HGH000072C39390GE32021',
   '20NU2HGH000077EBOLCV9390GUA2021'
]


Mozambique_grants = [
   '23NU2GGH0024629390FKR2023', '20NU2HGH000051C69390JFD2022', '20NU2HGH000051C39390GBN2022', '21NU2GGH00237222C39390GAP2022', '21NU2GGH002372PEC69390J6U2022', '22NU2GGH0024019390FKR2023', '20NU2HGH0000519390K202022'
]


Global_grants= [
   '20NU2GGH0023029390FKV2021', '21NU2GGH00235022PEC69390J6V2022', '19NU2GGH00219423HOP9390HEC2022',
   '19NU2GGH00219419HQTB9390FKV2020', '21NU2GGH0023789390K872023', '22NU2GGH0024369390HD32022', '20NU2HGH000006C39390GAM2021', '20NU2GGH00231221C39390G802021', '21NU2HGH000088C39390GDP2021',
   '20NU2HGH0000049390EQL2020', '20NU2HGH00000693906BX2021', '20NU2HGH000077C39390GFF2021', '20NU2HGH000072CV9390ETK2020', '20NU2HGH000072C39390GE32021',
   '20NU2HGH000077EBOLCV9390GUA2021', '23NU2GGH0024629390FKR2023', '20NU2HGH000051C69390JFD2022', '20NU2HGH000051C39390GBN2022', '21NU2GGH00237222C39390GAP2022', '21NU2GGH002372PEC69390J6U2022', '22NU2GGH0024019390FKR2023', '20NU2HGH0000519390K202022'
]


BAC_grants = ['NU2GGH0024062022', 'NU2HGH0000982021', 'NU2HGH0000042021', 'U01GH0022482020', 'NU2GGH0024062022', 'NU2GGH0022152020', 'NU51IP0009422022', 'U01GH0023382021', 'NU2GGH0021942020', 'NU2GGH0021712021', 'NU2HGH0000742020', 'NU2GGH0024052023', 'NU2GGH0024272023', 'NU2GGH0023582023', 'NU2GGH0023782023', 'NU2GGH0022982023', 'NU2HGH0000812021', 'NU2HGH0000382023', 'U01GH0022482019', 'NU2GGH0022212023']


BAC_SA_Grants = ['NU2GGH0021892020',
'NU2GGH0021952020',
'NU2GGH0021902020',
'NU2GGH0021932020',
'NU2GGH0019372020',
'NU2GGH0021942020',
'NU2GGH0021882020',
'NU2GGH0019372021',
'NU2GGH0019802020']


BAC_US_Grants = ['NU14GH0012382020', 'NU2GGH0024062022', 'NU2HGH0000982021', 'NU51IP0009422022', 'NU2HGH0001002021', 'NU2HGH0000052021', 'NU2GGH0023212021', 'NU50CK0005472021', 'NU14GH0012382020', 'NU50CK0004942020', ]


BAC_grant_tool = ['NU14GH0012382020', 'U01GH0022482020',
   #     'NU2GGH0013532020', 'NU2GGH0014632019',
   #    'NU2GGH0019372020', 'NU2GGH0019372021', 'NU2GGH0019762020',
   #    'NU2GGH0019782020', 'NU2GGH0019792020', 'NU2GGH0019802020',
   #    'NU2GGH0019992020', 'NU2GGH0020002022', 'NU2GGH0020022022',
   #    'NU2GGH0020082020', 'NU2GGH0020102020', 'NU2GGH0020212020',
   #    'NU2GGH0020222020', 'NU2GGH0020272020', 'NU2GGH0020462020',
   #    'NU2GGH0020592021', 'NU2GGH0020902020'
      ]


@app.route("/")
def index():
   return render_template("index.html")


@app.route("/Help")
def help():
   return render_template("help.html")


@app.route("/AI", methods=["GET", "POST"])
def ai_summary():
   if request.method == "POST":
       request_text = request.form.get("summaryInput")
       summary = ai_summary(request_text)
       #print(summary)
       return render_template("ai_output.html", summary = summary)
   return render_template("ai_input.html")


@app.route("/generate-summary", methods=["GET", "POST"])
def generate_summary():
   try:
       data = request.get_json()
       prompt = data.get("prompt")


       summary = ai_summary(prompt)


       return jsonify({"summary": summary})
   except Exception as e :
       return jsonify({"error": str(e)}), 500


@app.route("/generate-summary-grant", methods=["GET", "POST"])
def generate_summary_grant():
   try:
       data = request.get_json()
       prompt = data.get("prompt")


       summary = ai_summary_grant(prompt)


       return jsonify({"summary": summary})
   except Exception as e :
       return jsonify({"error": str(e)}), 500


@app.route("/grant_old", methods=["GET", "POST"])
def grant_old():
   if request.method == "POST":
       grant_name = request.form.get("grant_name")
       return generate_graph(grant_name)
   return render_template("grant.html", grants=grants)


@app.route("/grant", methods=["GET", "POST"])
def grant():
   if request.method == "POST":
       grant_name = request.form.get("grant_name")
       #need to implement method to grab country via grant_name
       data, grant_data, grant_name2, grant_months_remaining, grantee, grant_obligation, grant_liquidated, grant_udo, udo_percentage, country = generate_graph_with_grant(grant_name)
       #print(country)
       #country = "UGANDA"
       #grantee = "Uganda Medical"
       country_area_data, avg_line = generate_country_graph_without_overlay(country)
       #print("Country Area Data", country_area_data)


       #print("Grant Data:", grant_data)


       forecast_data = run_regression_model(grant_name, grant_obligation, grant_months_remaining)


       #print("Forecast Data", forecast_data)


       #print(avg_line)
       return render_template("grant_chart3.html", data=data, grant_data=grant_data, grant_name = grant_name2, months_remaining = grant_months_remaining, country_area_data = country_area_data, avg_line = avg_line, grantee = grantee, grant_obligation = grant_obligation, grant_liquidated = grant_liquidated, grant_udo = grant_udo, udo_percentage = udo_percentage, country = country, forecast_data = forecast_data)
   return render_template("grant_js_form.html", grants=BAC_grant_tool)


@app.route("/global_portfolio")
def portfolio_global():
   #data = generate_graph_without_overlay()
   country = "GLOBAL"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   print("Latest month data", latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   #save latest_months_data to excel
   #latest_months_data_df = pd.DataFrame(latest_months_data)
   #latest_months_data_df.to_excel("latest_months_data.xlsx")
   #print(area_data)


   #get the lenght of BAC_grants
   total_grants = len(BAC_grants)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "Global", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants, total_grants = total_grants)


@app.route("/api/global_portfolio")
def api_portfolio_global():
   country = "GLOBAL"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   total_grants = len(BAC_grants)
   remaining_obligations = total_obligations - total_liquidated


   return jsonify({
       "area_data": area_data,
       "latest_months_data": latest_months_data,
       "total_obligations": total_obligations,
       "total_liquidated": total_liquidated,
       "remaining_obligations": remaining_obligations,
       "total_current_UDO": total_current_UDO,
       "UDO_percentage": UDO_percentage,
       "country": country,
       "num_grants": num_grants,
       "total_grants": total_grants,
       "country_area_data": country_area_data,
       "avg_line": avg_line
   })


@app.route("/SA")
def portfolio_SA():
   #data = generate_graph_without_overlay()
   country = "SOUTH AFRICA"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_SA_Grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "South Africa", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)


BAC_UGANDA_Grants = [
# 'NU2GGH0013532020',
# 'NU2GGH0020022022',
# 'NU2GGH0020222020',
# 'NU2GGH0020462020',
# 'NU2GGH0021402020',
'NU2GGH0023092020',
'NU2GGH0023562021',
'NU2HGH0000342020',
'NU2HGH0000452020',
'NU2HGH0000462020',
'NU66GH0021722020',
'NU2GGH0013532020',
# 'NU2GGH0020222020',
# 'NU2GGH0020462020',
'NU2GGH0020022022',
# 'NU2GGH0021402020',
'U01GH0022482020']


BAC_UGANDA_SELECTED_IN_PROGRESS = ['U01GH0022482020',  'NU2GGH0023582023',  'NU2HGH0000382023', 'U01GH0022482019', 'NU2GGH0022212023']


@app.route("/Uganda")
def portfolio_Uganda():
   #data = generate_graph_without_overlay()
   country = "UGANDA"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_UGANDA_SELECTED_IN_PROGRESS)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   total_grants = len(BAC_UGANDA_Grants)




   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "Uganda", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants, total_grants = total_grants)


@app.route("/api/uganda_portfolio", methods=["GET"])
def api_portfolio_uganda():
   country = "UGANDA"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_UGANDA_SELECTED_IN_PROGRESS)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)


   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]


   total_grants = len(BAC_UGANDA_Grants)
   remaining_obligations = total_obligations - total_liquidated


   return jsonify({
       "area_data": area_data,
       "latest_months_data": latest_months_data,
       "total_obligations": total_obligations,
       "total_liquidated": total_liquidated,
       "remaining_obligations": remaining_obligations,
       "total_current_UDO": total_current_UDO,
       "UDO_percentage": UDO_percentage,
       "country": "Uganda",
       "country_area_data": country_area_data,
       "avg_line": avg_line,
       "num_grants": num_grants,
       "total_grants": total_grants
   })


@app.route("/US")
def portfolio_US():
   #data = generate_graph_without_overlay()
   country = "UNITED STATES"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(BAC_US_Grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "United States", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)


@app.route("/India")
def portfolio_India():
   #data = generate_graph_without_overlay()
   country = "INDIA"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(India_grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "India", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)


@app.route("/Ethiopia")
def portfolio_Ethiopia():
   #data = generate_graph_without_overlay()
   country = "ETHIOPIA"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(Ethiopia_grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "Ethiopia", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)


@app.route("/Mozambique")
def portfolio_Mozambique():
   #data = generate_graph_without_overlay()
   country = "MOZAMBIQUE"
   area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(Mozambique_grants)
   country_area_data, avg_line = generate_country_graph_without_overlay(country)
   #print(latest_months_data)
   if isinstance(area_data, tuple):
       return area_data[0], area_data[1]
   #return render_template("SA.html", data=data)
   # Render the template with the generated data


   # Formatting
   #total_obligations = "${:,.0f}".format(total_obligations)
   #total_liquidated = "${:,.0f}".format(total_liquidated)


   remaining_obligations = total_obligations - total_liquidated


   return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "Mozambique", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)


#function to get the target time series
def get_target_time_series(grant_to_forecast):
   import joblib


   # Load the scaled_grant_series dictionary from the file
   scaled_grant_series = joblib.load('scaled_grant_series.pkl')


   # Initialize dictionaries for train and validation sets
   train_grant_series = []
   val_grant_series = []


   # Initialize variable to store the target time series
   target_time_series = None
   target_time_series_val = None


   count = 0


   # Loop through each series in grant_dfs
   for grant_id, grant_series in scaled_grant_series.items():


       # Check if the series length is at least 12
       if len(grant_series) >= 24:


           count+=1


           # Split the series into train and validation sets
           train_one, val_one = grant_series[:-12], grant_series[-12:]


           # Store the split data in the respective lists
           train_grant_series.append(train_one)
           val_grant_series.append(val_one)


           # Check if the current grant_id matches the target grant_id
           if grant_id == grant_to_forecast:
               target_time_series = train_one
               target_time_series_val = val_one
               target_time_series_combined = grant_series


  
   #return target_time_series,
   return target_time_series_combined




#Create function to use darts to load the NBEATS_2Epoch
def run_model(grant_to_forecast):
   from darts.models import NBEATSModel


   model_name = "NBEATS_2Epoch"


   model_one_two = NBEATSModel.load_from_checkpoint(model_name=model_name, best=False)


   #print grant_to_forecast to console
   print("Grant to Forecast: ", grant_to_forecast)


   target_time_series = get_target_time_series(grant_to_forecast)


   pred = model_one_two.predict(n=24, series=target_time_series)


   # Altering Line
   highest_pred = float('-inf')  # Initialize to negative infinity
   new_values = []


   for value in pred.values():
       if value > highest_pred:
           highest_pred = value
       new_values.append(highest_pred)


   # Create a new TimeSeries with the modified values
   pred = pred.with_values(new_values)


   pred = scale_preds(pred, grant_to_forecast, target_time_series)


   forecast_data = {
       'GrantTimeElapsed': pred.index.tolist(),
       'ObligationSpent': pred['Obligation Spent'].tolist(),
   }


   print(forecast_data)


   #return nothing
   return forecast_data


def scale_preds(pred, grant_to_forecast, target_time_series):
   import joblib
   # Load the scalers dictionary from the file
   scalers = joblib.load('scalers.pkl')
   grant_dfs = joblib.load('grant_dfs.pkl')


   # Retrieve the scaler for a specific grant ID
   grant_id = grant_to_forecast
   scaler = scalers[grant_id]


   # Use the scaler to unscale data
   unscaled_data = scaler.inverse_transform(pred)




   # Normalize data
   #unscaled_data = unscaled_data / 17200000 * 100
   # Normalize the actual data
   grant_dfs_selected = grant_dfs[grant_to_forecast]


   # Replace dates with index values
   pred_df = unscaled_data.pd_dataframe().reset_index(drop=True)
   scaled_grant_series_df = grant_dfs_selected.pd_dataframe().reset_index(drop=True)


   #get last
   target_time_series = scaler.inverse_transform(target_time_series)
   target_time_series = target_time_series.pd_dataframe().reset_index(drop=True)


   print("TARGET:",target_time_series)


   # Increase the indexes of pred by the last index of the scaled_grant_series_df
   last_index = target_time_series.index[-1]


   pred_df.index = pred_df.index + scaled_grant_series_df.index[-1] + 12
  
   # Divide x-axis by 60 to get percent
   pred_df.index = pred_df.index / 60 * 100
   scaled_grant_series_df.index = scaled_grant_series_df.index / 60 * 100


   #print the type of pred_df
   #print(type(pred_df))


   return pred_df


def run_regression_model(grant_to_forecast, grant_obligation, grant_months_remaining):
   import joblib


   # Load the scaled_grant_series dictionary from the file
   series_dict_diff = joblib.load('series_dict_diff.pkl')
   scalers_dict_diff = joblib.load('scalers_dict_diff.pkl')


   model = XGBModel(
       lags=6, output_chunk_length=1, likelihood="quantile", quantiles=[0.05, 0.5, 0.95]
   )


   model.fit(series_dict_diff[grant_to_forecast])


   pred_samples = model.predict(series = series_dict_diff[grant_to_forecast],n=24, num_samples=500)


   #Unscale the results
   back_diff = scalers_dict_diff[grant_to_forecast].inverse_transform(series_dict_diff[grant_to_forecast])


   back_pred_samples = scalers_dict_diff[grant_to_forecast].inverse_transform(pred_samples)


   #Convert back_diff to cumsum
   back_diff = back_diff.cumsum()


   #Covert all the columns of back_pred_samples to cumsum
   back_pred_samples = back_pred_samples.cumsum() + back_diff[-1]


   #Conmver y-axis to index values
   back_pred_samples = back_pred_samples.pd_dataframe().reset_index(drop=True)


   #Conmver y-axis to index values
   back_diff = back_diff.pd_dataframe().reset_index(drop=True)


   #Divide x-axis by 60 to get percent
   back_diff.index = back_diff.index / 60 * 100
  
   # Divide x-axis by 60 to get percent
   back_pred_samples.index = back_pred_samples.index / 60 * 100


   #Divide y-axis by grant_obligation
   back_pred_samples = back_pred_samples['Disbursement_s0'] / grant_obligation * 100


   #Print back_pred_samples
   print(back_pred_samples)


   #grant time elapsed
   time_elapsed = ((61 - grant_months_remaining) / 60 ) * 100


   #Increase back_pred_samples x-axis by time_elapsed
   back_pred_samples.index = back_pred_samples.index + time_elapsed




   forecast_data = {
       'GrantTimeElapsed': back_pred_samples.index.tolist(),
       'ObligationSpent': back_pred_samples.tolist(),
   }


   # Remove keys and values where GrantTimeElapsed is higher than 100
   filtered_data = {
       'GrantTimeElapsed': [],
       'ObligationSpent': []
   }
   for time, spent in zip(forecast_data['GrantTimeElapsed'], forecast_data['ObligationSpent']):
       if time <= 100:
           filtered_data['GrantTimeElapsed'].append(time)
           filtered_data['ObligationSpent'].append(spent)


   #print("FORECAST:", filtered_data)


   return filtered_data








def ai_summary(docs):
   try:
       from langchain.chains.combine_documents import create_stuff_documents_chain
       from langchain.chains.llm import LLMChain
       from langchain_core.prompts import ChatPromptTemplate


       template = "Based on an analysis of 22 grants, 3 grants fall within the red shaded area, indicating they will likely have a large unliquidated balance and may require intervention from project officers or program leaders. The red shaded area, which represents grants with less than 100% percent liquidation pattern, is calculated using all grant data from FY XX to FY YY to identify liquidation trends. Notably, 19 out of the 22 grants exhibit a normal liquidation rate and are expected to close with minimal or no unliquidated balance. Grants obligations older than 2.5 years require proactive engagement to facilitate efficient liquidation or de-obligation."


       # Define prompt
       prompt = ChatPromptTemplate.from_messages(
           [("system", "You are a graph analyzer assistant. You will be provided a graph that has __ points that represent the latest Percent Obligation Spent across the Percent Elapsed Time of grants. These points are different color dots that each represent a unique grant. The red line with a red shaded area below is the area trend of Grants with Less Than %100 Percent Liquidation Pattern, this pattern indicates that these grants are expected to leave behind obligation which requires intervention. If a grant falls within/below this red shaded area, these grants need intervention. Make sure to mention how the Red Shaded Areas are being calculated. Use a single paragraph format. \\n Write a concise summary using this data:\\n\\n{context}. \\n Reference this example summary: \\n\\n{template}")]
       )


       # Instantiate chain
       chain = create_stuff_documents_chain(llm, prompt)


       # Convert input text to the expected format
       document_objects = [Document(content=docs)]


       # Invoke chain
       result = chain.invoke({"context": document_objects, "template": template})
       return result


   except Exception as e:
       return str(e), 500
  
def ai_summary_grant(docs):
   try:
       from langchain.chains.combine_documents import create_stuff_documents_chain
       from langchain.chains.llm import LLMChain
       from langchain_core.prompts import ChatPromptTemplate


       # Define prompt
       prompt = ChatPromptTemplate.from_messages(
           [("system", "Referencing the following: 'You are a graph analyzer assistant. You will be provided a graph with Grant Trend Data representing the historical percent ObligationSpent against the year GrantTimeElapsed (Y1, Y2, Y3, Y4) for a specific grant. The red line with a red shaded area below is the area trend of Grants with Less Than %100 Percent Liquidation Pattern, this pattern indicates that these grants are expected to leave behind obligation which requires intervention. Simply compare the Last Point in Grant Trend Data to the corresponding Red Shaded Area Data with the closest similar Time Elapsed to identify if the Grant Trend Data's latest point's Obligation Spent falls above or below the Red Shaded Area latest point's Obligation Spent. Make sure to mention how the Red Shaded Areas are being calculated using either global or country specific data. Use a single paragraph format.' Write a concise summary using this data:\\n\\n{context} Make sure to accurately identify the x-axis value of the latest point and the corresponding y-axis value. Never say: 'x-axis value of _', instead say 'Year _'. Make sure to use all lower case expect for beginning of sentences and Year _. Replace Grant Trend Data with grant actual liquidation, Percent Obligation Spent with liquidated obligation percentage. Round all percentages to whole numbers.")]
       )


       # Instantiate chain
       chain = create_stuff_documents_chain(llm, prompt)


       # Convert input text to the expected format
       document_objects = [Document(content=docs)]


       # Invoke chain
       result = chain.invoke({"context": document_objects})
       return result


   except Exception as e:
       return str(e), 500


def latest_months_in_grants(grants):
   try:
       import pandas as pd


       # Load the BAC_Data.xlsx file
       bac_data = pd.read_excel('BAC_Data.xlsx')


       # Process data
       obligation_progression = bac_data[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Grantee", "Country"]]
       obligation_progression["Month"] = pd.to_datetime(obligation_progression["Month"], infer_datetime_format=True)
       obligation_progression["Grant End Date"] = pd.to_datetime(obligation_progression["Grant End Date"], infer_datetime_format=True)
       obligation_progression["Grant End Date EOM"] = obligation_progression["Grant End Date"] + pd.offsets.MonthEnd(0)
       obligation_progression["Grant Start Date EOM"] = obligation_progression["Grant Start Date"] + pd.offsets.MonthEnd(0)
       obligation_progression = obligation_progression[obligation_progression["Month"] <= obligation_progression["Grant End Date EOM"]]


       def month_diff(start, end):
           return (end.year - start.year) * 12 + end.month - start.month
      
       # Calculate grant length in number of months
       obligation_progression["Grant Length Months"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Grant End Date EOM"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Elapsed"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Month"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months remaining in the grant
       obligation_progression["Months Remaining"] = obligation_progression["Grant Length Months"] - obligation_progression["Grant Months Elapsed"]
      
       # Calculate percent of grant time elapsed
       obligation_progression["Grant Time Elapsed"] = obligation_progression["Grant Months Elapsed"] / obligation_progression["Grant Length Months"]


       # Calculate percent of obligation spent
       obligation_progression["Obligation Spent"] = obligation_progression["Disbursement"] / obligation_progression["Obligation"]


       # Filter out rows with Obligation Spent greater than 1
       obligation_progression = obligation_progression[obligation_progression["Obligation Spent"] <= 1]


       # Filter out rows with Grant Time Elapsed less than 0
       obligation_progression = obligation_progression[obligation_progression["Grant Time Elapsed"] >= 0]


       # Convert to percentages
       obligation_progression["Grant Time Elapsed"] *= 100
       obligation_progression["Obligation Spent"] *= 100


       # Ensure Obligation Spent is between 0 and 100
       obligation_progression = obligation_progression[(obligation_progression["Obligation Spent"] >= 0) & (obligation_progression["Obligation Spent"] <= 100)]


       udo_progression = obligation_progression[obligation_progression["UDO Status"] == "UDO"]
       non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non UDO"]


       # Train model for UDO
       X = udo_progression[["Grant Time Elapsed"]]
       y = udo_progression["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_udo = LinearRegression()
       model_udo.fit(X_train, y_train)
       y_pred = model_udo.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)


       # Train model for Non-UDO
       non_udo_progression_clean = non_udo_progression.replace([np.inf, -np.inf], np.nan).dropna()
       X = non_udo_progression_clean[["Grant Time Elapsed"]]
       y = non_udo_progression_clean["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_nonudo = LinearRegression()
       model_nonudo.fit(X_train, y_train)
       y_pred = model_nonudo.predict(X_test)


       # Generate predictions
       continuous_range = np.arange(0.00, 100.0, 0.1)  # Adjusted range for percentages
       disbursement_predictions = pd.DataFrame({'Grant Time Elapsed': continuous_range})
       udo_pred = model_udo.predict(disbursement_predictions[['Grant Time Elapsed']])
       non_udo_pred = model_nonudo.predict(disbursement_predictions[['Grant Time Elapsed']])


       # Clip predictions to ensure they are between 0 and 100
       udo_pred = np.clip(udo_pred, 0, 100)
       non_udo_pred = np.clip(non_udo_pred, 0, 100)


       disbursement_predictions['UDO Predicted Level'] = udo_pred
       disbursement_predictions['Non UDO Predicted Level'] = non_udo_pred


       # Convert the data to JSON format
       area_data = {
           'GrantTimeElapsed': disbursement_predictions['Grant Time Elapsed'].tolist(),
           'UDOPredictedLevel': disbursement_predictions['UDO Predicted Level'].tolist(),
           'NonUDOPredictedLevel': disbursement_predictions['Non UDO Predicted Level'].tolist()
       }
      
       # Gather Latest Months
       latest_months_all_indicies = obligation_progression.groupby('Unique ID')['Month'].idxmax()
       latest_months_all_rows = obligation_progression.loc[latest_months_all_indicies]


       # Filter to only include the specified grants
       latest_months_filtered = latest_months_all_rows[latest_months_all_rows['Unique ID'].isin(grants)]


       # Convert the data to JSON format for Chart.js
       data = {
           'UniqueID': latest_months_filtered['Unique ID'].tolist(),
           'GrantTimeElapsed': latest_months_filtered['Grant Time Elapsed'].tolist(),
           'ObligationSpent': latest_months_filtered['Obligation Spent'].tolist(),
           'MonthsRemaining': latest_months_filtered['Months Remaining'].tolist(),
           'Grantee': latest_months_filtered['Grantee'].tolist(),
           'Country': latest_months_filtered['Country'].tolist(),
       }


       #print(data)


       # Calculate Total Obligations in Dollars using the latest_months_filtered
       total_obligations = latest_months_filtered["Obligation"].sum()


       # Calculate Total Liquidated in Dollars using the latest_months_filtered
       total_liquidated = latest_months_filtered["Disbursement"].sum()


       # Calculate Current UDO using Difference between Total Obligations and Total Liquidated
       total_current_UDO = total_obligations - total_liquidated


       # Calculate UDO percentage using current udo over total obligations
       UDO_percentage = (total_current_UDO / total_obligations) * 100


       # Number of grants
       num_grants = len(latest_months_filtered)


       return area_data, data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants


   except Exception as e:
       return str(e), 500


def generate_graph_with_grant(grant_name):
   try:
       # Load the BAC_Data.xlsx file
       bac_data = pd.read_excel('BAC_Data.xlsx')


       # Process data
       obligation_progression = bac_data[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Grantee", "Country"]]


       obligation_progression["Month"] = pd.to_datetime(obligation_progression["Month"], infer_datetime_format=True)
       obligation_progression["Grant End Date"] = pd.to_datetime(obligation_progression["Grant End Date"], infer_datetime_format=True)
       obligation_progression["Grant End Date EOM"] = obligation_progression["Grant End Date"] + pd.offsets.MonthEnd(0)
       obligation_progression["Grant Start Date EOM"] = obligation_progression["Grant Start Date"] + pd.offsets.MonthEnd(0)
       obligation_progression = obligation_progression[obligation_progression["Month"] <= obligation_progression["Grant End Date EOM"]]
      
       def month_diff(start, end):
           return (end.year - start.year) * 12 + end.month - start.month
      
       # Calculate grant length in number of months
       obligation_progression["Grant Length Months"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Grant End Date EOM"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Elapsed"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Month"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate percent of grant time elapsed
       obligation_progression["Grant Time Elapsed"] = obligation_progression["Grant Months Elapsed"] / obligation_progression["Grant Length Months"]


       # Calculate percent of obligation spent
       obligation_progression["Obligation Spent"] = obligation_progression["Disbursement"] / obligation_progression["Obligation"]


       # Filter out rows with Obligation Spent greater than 1
       obligation_progression = obligation_progression[obligation_progression["Obligation Spent"] <= 1]


       # Filter out rows with Grant Time Elapsed less than 0
       obligation_progression = obligation_progression[obligation_progression["Grant Time Elapsed"] >= 0]


       # Convert to percentages
       obligation_progression["Grant Time Elapsed"] *= 100
       obligation_progression["Obligation Spent"] *= 100


       # Ensure Obligation Spent is between 0 and 100
       obligation_progression = obligation_progression[(obligation_progression["Obligation Spent"] >= 0) & (obligation_progression["Obligation Spent"] <= 100)]


       udo_progression = obligation_progression[obligation_progression["UDO Status"] == "UDO"]
       non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non UDO"]


       # Train model for UDO
       X = udo_progression[["Grant Time Elapsed"]]
       y = udo_progression["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_udo = LinearRegression()
       model_udo.fit(X_train, y_train)
       y_pred = model_udo.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)


       # Train model for Non-UDO
       non_udo_progression_clean = non_udo_progression.replace([np.inf, -np.inf], np.nan).dropna()
       X = non_udo_progression_clean[["Grant Time Elapsed"]]
       y = non_udo_progression_clean["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_nonudo = LinearRegression()
       model_nonudo.fit(X_train, y_train)
       y_pred = model_nonudo.predict(X_test)


       # Generate predictions
       continuous_range = np.arange(0.00, 100.0, 0.1)  # Adjusted range for percentages
       disbursement_predictions = pd.DataFrame({'Grant Time Elapsed': continuous_range})
       udo_pred = model_udo.predict(disbursement_predictions[['Grant Time Elapsed']])
       non_udo_pred = model_nonudo.predict(disbursement_predictions[['Grant Time Elapsed']])


       # Clip predictions to ensure they are between 0 and 100
       udo_pred = np.clip(udo_pred, 0, 100)
       non_udo_pred = np.clip(non_udo_pred, 0, 100)


       disbursement_predictions['UDO Predicted Level'] = udo_pred
       disbursement_predictions['Non UDO Predicted Level'] = non_udo_pred


       # Convert the data to JSON format
       data = {
           'GrantTimeElapsed': disbursement_predictions['Grant Time Elapsed'].tolist(),
           'UDOPredictedLevel': disbursement_predictions['UDO Predicted Level'].tolist(),
           'NonUDOPredictedLevel': disbursement_predictions['Non UDO Predicted Level'].tolist()
       }


       if grant_name:
           grant_data = obligation_progression[obligation_progression['Unique ID'] == grant_name]


       latest_month_for_grant = grant_data.sort_values(by='Month', ascending = False).iloc[0]


       grant_months_remaining = latest_month_for_grant['Grant Length Months'] - latest_month_for_grant['Grant Months Elapsed']
      
       grant_obligation = latest_month_for_grant['Obligation']
       grant_grantee = latest_month_for_grant['Grantee']
       grant_liquidated = latest_month_for_grant['Disbursement']
       grant_udo = grant_obligation - grant_liquidated
       grant_UDO_percentage = (grant_udo / grant_obligation) * 100
       grant_country = latest_month_for_grant['Country']


       grant_data = {
           'GrantTimeElapsed': grant_data['Grant Time Elapsed'].tolist(),
           'ObligationSpent': grant_data['Obligation Spent'].tolist(),
       }
       return data, grant_data, grant_name, grant_months_remaining, grant_grantee, grant_obligation, grant_liquidated, grant_udo, grant_UDO_percentage, grant_country




   except Exception as e:
       return str(e), 500


def generate_graph_without_overlay():
   try:
       # Load Excel files
       udo_ts = pd.read_excel("GHC FY21-23 Grant UDO Data.xlsx")
       udo_c = pd.read_excel("GHC Grant Data Test.xlsx", skiprows=3, header=1)
      
       # Select relevant columns
       udo_c_selected = udo_c[["Unique ID", "UDO Status", "Recoverable", "Grant Start Date", "Grant End Date"]]
      
       # Merge dataframes
       udo_combined = udo_ts.merge(udo_c_selected, how='left', on='Unique ID')
       udo_combined.sort_values(by=['Unique ID', 'Month'], axis=0, inplace=True, ignore_index=True)
      
       # Process data
       obligation_progression = udo_combined[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Recoverable"]]
       obligation_progression["Month"] = pd.to_datetime(obligation_progression["Month"], infer_datetime_format=True)
       obligation_progression["Grant End Date"] = pd.to_datetime(obligation_progression["Grant End Date"], infer_datetime_format=True)
       obligation_progression["Grant End Date EOM"] = obligation_progression["Grant End Date"] + pd.offsets.MonthEnd(0)
       obligation_progression["Grant Start Date EOM"] = obligation_progression["Grant Start Date"] + pd.offsets.MonthEnd(0)
       obligation_progression = obligation_progression[obligation_progression["Month"] <= obligation_progression["Grant End Date EOM"]]
      
       def month_diff(start, end):
           return (end.year - start.year) * 12 + end.month - start.month
      
       # Calculate grant length in number of months
       obligation_progression["Grant Length Months"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Grant End Date EOM"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Elapsed"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Month"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate percent of grant time elapsed
       obligation_progression["Grant Time Elapsed"] = obligation_progression["Grant Months Elapsed"] / obligation_progression["Grant Length Months"]


       # Calculate percent of obligation spent
       obligation_progression["Obligation Spent"] = obligation_progression["Disbursement"] / obligation_progression["Obligation"]


       # Filter out rows with Obligation Spent greater than 1
       obligation_progression = obligation_progression[obligation_progression["Obligation Spent"] <= 1]


       # Filter out rows with Grant Time Elapsed less than 0
       obligation_progression = obligation_progression[obligation_progression["Grant Time Elapsed"] >= 0]


       # Convert to percentages
       obligation_progression["Grant Time Elapsed"] *= 100
       obligation_progression["Obligation Spent"] *= 100


       # Ensure Obligation Spent is between 0 and 100
       obligation_progression = obligation_progression[(obligation_progression["Obligation Spent"] >= 0) & (obligation_progression["Obligation Spent"] <= 100)]


       udo_progression = obligation_progression[obligation_progression["UDO Status"] == "ULO"]
       non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non ULO"]


       # Train model for UDO
       X = udo_progression[["Grant Time Elapsed"]]
       y = udo_progression["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_udo = LinearRegression()
       model_udo.fit(X_train, y_train)
       y_pred = model_udo.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)


       # Train model for Non-UDO
       non_udo_progression_clean = non_udo_progression.replace([np.inf, -np.inf], np.nan).dropna()
       X = non_udo_progression_clean[["Grant Time Elapsed"]]
       y = non_udo_progression_clean["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_nonudo = LinearRegression()
       model_nonudo.fit(X_train, y_train)
       y_pred = model_nonudo.predict(X_test)


       # Generate predictions
       continuous_range = np.arange(0.00, 100.0, 0.1)  # Adjusted range for percentages
       disbursement_predictions = pd.DataFrame({'Grant Time Elapsed': continuous_range})
       udo_pred = model_udo.predict(disbursement_predictions[['Grant Time Elapsed']])
       non_udo_pred = model_nonudo.predict(disbursement_predictions[['Grant Time Elapsed']])


       # Clip predictions to ensure they are between 0 and 100
       udo_pred = np.clip(udo_pred, 0, 100)
       non_udo_pred = np.clip(non_udo_pred, 0, 100)


       disbursement_predictions['UDO Predicted Level'] = udo_pred
       disbursement_predictions['Non UDO Predicted Level'] = non_udo_pred


       # Convert the data to JSON format
       data = {
           'GrantTimeElapsed': disbursement_predictions['Grant Time Elapsed'].tolist(),
           'UDOPredictedLevel': disbursement_predictions['UDO Predicted Level'].tolist(),
           'NonUDOPredictedLevel': disbursement_predictions['Non UDO Predicted Level'].tolist()
       }


       return data


   except Exception as e:
       return str(e), 500


def generate_country_graph_without_overlay(Country_Name):
   try:
       # Load the BAC_Data.xlsx file
       bac_data = pd.read_excel('BAC_Data.xlsx')


       # Process data
       obligation_progression = bac_data[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Grantee", "Country"]]


       obligation_progression["Month"] = pd.to_datetime(obligation_progression["Month"], infer_datetime_format=True)
       obligation_progression["Grant End Date"] = pd.to_datetime(obligation_progression["Grant End Date"], infer_datetime_format=True)
       obligation_progression["Grant End Date EOM"] = obligation_progression["Grant End Date"] + pd.offsets.MonthEnd(0)
       obligation_progression["Grant Start Date EOM"] = obligation_progression["Grant Start Date"] + pd.offsets.MonthEnd(0)
       obligation_progression = obligation_progression[obligation_progression["Month"] <= obligation_progression["Grant End Date EOM"]]
              
       def month_diff(start, end):
           return (end.year - start.year) * 12 + end.month - start.month
              
               # Calculate grant length in number of months
       obligation_progression["Grant Length Months"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Grant End Date EOM"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Elapsed"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Month"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Left"] = obligation_progression["Grant Length Months"] - obligation_progression["Grant Months Elapsed"]


       # Calculate percent of grant time elapsed
       obligation_progression["Grant Time Elapsed"] = obligation_progression["Grant Months Elapsed"] / obligation_progression["Grant Length Months"]


       # Calculate percent of obligation spent
       obligation_progression["Obligation Spent"] = obligation_progression["Disbursement"] / obligation_progression["Obligation"]


       # Filter out rows with Obligation Spent greater than 1
       obligation_progression = obligation_progression[obligation_progression["Obligation Spent"] <= 1]


       # Filter out rows with Grant Time Elapsed less than 0
       obligation_progression = obligation_progression[obligation_progression["Grant Time Elapsed"] >= 0]


       # Convert to percentages
       obligation_progression["Grant Time Elapsed"] *= 100
       obligation_progression["Obligation Spent"] *= 100


       # Ensure Obligation Spent is between 0 and 100
       obligation_progression = obligation_progression[(obligation_progression["Obligation Spent"] >= 0) & (obligation_progression["Obligation Spent"] <= 100)]


       uid_counts = obligation_progression['Unique ID'].value_counts()


       # Step 2: Filter UIDs with more than 2 records
       uids_with_more_than_2_records = uid_counts[uid_counts > 1].index


       # Step 3: Filter the DataFrame to include only these UIDs
       filtered_obligation_progression = obligation_progression[obligation_progression['Unique ID'].isin(uids_with_more_than_2_records)]


       if Country_Name != 'GLOBAL':
           filtered_obligation_progression = filtered_obligation_progression[filtered_obligation_progression['Country'] == Country_Name]


       obligation_progression = filtered_obligation_progression


       #Change Grant Lenght Months to 60
       obligation_progression["Grant Length Months"] = 60


       # Ensure that the 'Grant Months Elapsed' does not exceed 60 months
       obligation_progression["Grant Months Elapsed"] = obligation_progression["Grant Months Elapsed"].apply(lambda x: 60 if x > 60 else x)


       udo_progression = obligation_progression[obligation_progression["UDO Status"] == "UDO"]
       non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non UDO"]


       # Filter out rows with UDO Status == "In-Progress"
       # obligation_progression_filtered = obligation_progression[obligation_progression["UDO Status"] != "In-Progress"]


       # Calculate the mean Obligation for each Month Elapsed and then convert to percent
       obligation_progression_avg = obligation_progression.groupby("Grant Months Elapsed")["Obligation Spent"].mean().reset_index()


       # Ensure the average line includes all months up to 60
       full_range = pd.DataFrame({'Grant Months Elapsed': range(61)})
       obligation_progression_avg = pd.merge(full_range, obligation_progression_avg, on="Grant Months Elapsed", how="left")
       obligation_progression_avg["Obligation Spent"] = obligation_progression_avg["Obligation Spent"].fillna(0)


       # Convert Grant Months Elapsed to percent by dividing by 60 and multiplying by 100
       obligation_progression_avg["Grant Months Elapsed"] = obligation_progression_avg["Grant Months Elapsed"] / 60 * 100


       avg_obligation_spent_list = {
           'GrantTimeElapsed': obligation_progression_avg['Grant Months Elapsed'].tolist(),
           'ObligationSpent': obligation_progression_avg['Obligation Spent'].tolist()
       }


       # Apply rolling window to smooth the data
       if Country_Name == 'GLOBAL':
           window = 6
       else:
           window = 12


       avg_obligation_spent_list['ObligationSpent'] = pd.Series(avg_obligation_spent_list['ObligationSpent']).rolling(window=window).mean().tolist()


       # Replace missing ObligationSpent values with 0
       avg_obligation_spent_list['ObligationSpent'] = [0 if math.isnan(x) else x for x in avg_obligation_spent_list['ObligationSpent']]


       # Train model for UDO
       X = udo_progression[["Grant Time Elapsed"]]
       y = udo_progression["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_udo = LinearRegression()
       model_udo.fit(X_train, y_train)
       y_pred = model_udo.predict(X_test)


       # Train model for Non-UDO
       non_udo_progression_clean = non_udo_progression.replace([np.inf, -np.inf], np.nan).dropna()
       X = non_udo_progression_clean[["Grant Time Elapsed"]]
       y = non_udo_progression_clean["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_nonudo = LinearRegression()
       model_nonudo.fit(X_train, y_train)
       y_pred = model_nonudo.predict(X_test)


       ## Generate predictions
       continuous_range = np.arange(0.00, 100.0, 0.1)  # Adjusted range for percentages
       disbursement_predictions = pd.DataFrame({'Grant Time Elapsed': continuous_range})
       udo_pred = model_udo.predict(disbursement_predictions[['Grant Time Elapsed']])
       non_udo_pred = model_nonudo.predict(disbursement_predictions[['Grant Time Elapsed']])


       # Clip predictions to ensure they are between 0 and 100
       udo_pred = np.clip(udo_pred, 0, 100)
       non_udo_pred = np.clip(non_udo_pred, 0, 100)


       disbursement_predictions['UDO Predicted Level'] = udo_pred
       disbursement_predictions['Non UDO Predicted Level'] = non_udo_pred


       # Convert the data to JSON format
       data = {
           'GrantTimeElapsed': disbursement_predictions['Grant Time Elapsed'].tolist(),
           'UDOPredictedLevel': disbursement_predictions['UDO Predicted Level'].tolist(),
           'NonUDOPredictedLevel': disbursement_predictions['Non UDO Predicted Level'].tolist()
       }


       return data, avg_obligation_spent_list


   except Exception as e:
       return str(e), 500


def generate_graph(grant_name):
   try:
       # Load Excel files
       udo_ts = pd.read_excel("GHC FY21-23 Grant UDO Data.xlsx")
       udo_c = pd.read_excel("GHC Grant Data Test.xlsx", skiprows=3, header=1)
      
       # Select relevant columns
       udo_c_selected = udo_c[["Unique ID", "UDO Status", "Recoverable", "Grant Start Date", "Grant End Date"]]
      
       # Merge dataframes
       udo_combined = udo_ts.merge(udo_c_selected, how='left', on='Unique ID')
       udo_combined.sort_values(by=['Unique ID', 'Month'], axis=0, inplace=True, ignore_index=True)
      
       # Process data
       obligation_progression = udo_combined[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Recoverable"]]
       obligation_progression["Month"] = pd.to_datetime(obligation_progression["Month"], infer_datetime_format=True)
       obligation_progression["Grant End Date"] = pd.to_datetime(obligation_progression["Grant End Date"], infer_datetime_format=True)
       obligation_progression["Grant End Date EOM"] = obligation_progression["Grant End Date"] + pd.offsets.MonthEnd(0)
       obligation_progression["Grant Start Date EOM"] = obligation_progression["Grant Start Date"] + pd.offsets.MonthEnd(0)
       obligation_progression = obligation_progression[obligation_progression["Month"] <= obligation_progression["Grant End Date EOM"]]
      
       def month_diff(start, end):
           return (end.year - start.year) * 12 + end.month - start.month
      
       # Calculate grant length in number of months
       obligation_progression["Grant Length Months"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Grant End Date EOM"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate months elapsed since grant start date
       obligation_progression["Grant Months Elapsed"] = obligation_progression.groupby("Unique ID").apply(
           lambda group: group.apply(
               lambda row: month_diff(row["Grant Start Date EOM"], row["Month"]), axis=1)).reset_index(level=0, drop=True)


       # Calculate percent of grant time elapsed
       obligation_progression["Grant Time Elapsed"] = obligation_progression["Grant Months Elapsed"] / obligation_progression["Grant Length Months"]


       # Calculate percent of obligation spent
       obligation_progression["Obligation Spent"] = obligation_progression["Disbursement"] / obligation_progression["Obligation"]


       # Filter out rows with Obligation Spent greater than 1
       obligation_progression = obligation_progression[obligation_progression["Obligation Spent"] <= 1]


       # Filter out rows with Grant Time Elapsed less than 0
       obligation_progression = obligation_progression[obligation_progression["Grant Time Elapsed"] >= 0]


       # Convert to percentages
       obligation_progression["Grant Time Elapsed"] *= 100
       obligation_progression["Obligation Spent"] *= 100


       # Ensure Obligation Spent is between 0 and 100
       obligation_progression = obligation_progression[(obligation_progression["Obligation Spent"] >= 0) & (obligation_progression["Obligation Spent"] <= 100)]


       udo_progression = obligation_progression[obligation_progression["UDO Status"] == "ULO"]
       non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non ULO"]


       # Train model for UDO
       X = udo_progression[["Grant Time Elapsed"]]
       y = udo_progression["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_udo = LinearRegression()
       model_udo.fit(X_train, y_train)
       y_pred = model_udo.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)


       # Train model for Non-UDO
       non_udo_progression_clean = non_udo_progression.replace([np.inf, -np.inf], np.nan).dropna()
       X = non_udo_progression_clean[["Grant Time Elapsed"]]
       y = non_udo_progression_clean["Obligation Spent"]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model_nonudo = LinearRegression()
       model_nonudo.fit(X_train, y_train)
       y_pred = model_nonudo.predict(X_test)


       # Generate predictions
       continuous_range = np.arange(0.00, 100.0, 0.1)  # Adjusted range for percentages
       disbursement_predictions = pd.DataFrame({'Grant Time Elapsed': continuous_range})
       udo_pred = model_udo.predict(disbursement_predictions[['Grant Time Elapsed']])
       non_udo_pred = model_nonudo.predict(disbursement_predictions[['Grant Time Elapsed']])


       # Clip predictions to ensure they are between 0 and 100
       udo_pred = np.clip(udo_pred, 0, 100)
       non_udo_pred = np.clip(non_udo_pred, 0, 100)


       disbursement_predictions['UDO Predicted Level'] = udo_pred
       disbursement_predictions['Non UDO Predicted Level'] = non_udo_pred


       # Plot the graph using the specified parameters
       ax = disbursement_predictions.plot(
           kind='area',
           x='Grant Time Elapsed',
           y=['UDO Predicted Level', 'Non UDO Predicted Level'],
           figsize=(10, 6),
           alpha=0.5,
           title='UDO & Non-UDO Disbursement Patterns',
           xlabel='X-Axis: % of Grant Time Elapsed',
           ylabel='% of Obligation Liquidated',
           grid=True,
           legend=False,  # Disable automatic legend
           stacked=True,
           color=['#99CCFF', '#FF9999']  # Swap colors for UDO and Non-UDO areas
       )


       # Manually create the legend
       custom_lines = [
           plt.Line2D([0], [0], color='#99CCFF', lw=4),
           plt.Line2D([0], [0], color='#FF9999', lw=4)
       ]
       ax.legend(custom_lines, ['Liquidation Pattern for Grants resulting in UDO', 'Liquidation Pattern for Grants with complete Liquidation'])


       # Overlay the specified grant's spent amount
       grant_status = "Grant not found"
       if grant_name:
           grant_data = obligation_progression[obligation_progression['Unique ID'] == grant_name]
           if not grant_data.empty:
               grant_data.plot(
                   x='Grant Time Elapsed',
                   y='Obligation Spent',
                   ax=ax,
                   label=f'{grant_name} Spent Amount',
                   linestyle='--',
                   color='black'  # Set grant prediction line to black
               )
               grant_status = f"Grant {grant_name} is {'UDO' if grant_data['UDO Status'].iloc[0] == 'ULO' else 'Non UDO'}"
               ax.legend(custom_lines + [plt.Line2D([0], [0], color='black', linestyle='--')],
                         ['Liquidation Pattern for Grants resulting in UDO', 'Liquidation Pattern for Grants with complete Liquidation', f'{grant_name} Spent Amount'])


       # Save the plot to a BytesIO object
       img = io.BytesIO()
       plt.savefig(img, format='png')
       img.seek(0)
       plt.close()


       # Encode the image to base64
       img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')


       # Render the template with the image and grant status
       return render_template("result.html", grant_status=grant_status, img_base64=img_base64)


   except Exception as e:
       return str(e), 500






if __name__ == '__main__':
   print("Starting flask")
   app.run(debug=True)




# In[ ]:





