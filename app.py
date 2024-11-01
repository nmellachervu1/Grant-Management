#!/usr/bin/env python
# coding: utf-8



from flask import Flask, request, render_template, jsonify
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import io
import base64

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set the matplotlib backend to 'Agg'
plt.switch_backend('Agg')

app = Flask(__name__)

# List of grants
grants = [
    '22R43GH0023699390JGK2022', '22R43GH0023899390JLL2022', '22R43GH002391CV9390JLL2022',
    '22R43GH002392CV9390JLL2022', '21NU2GGH0023399390FKN2021', '21NU2GGH0023399390HCW2021',
    '21R43GH00236793901A32021', '21R43GH002367939ZZMF2021', '21R43GH00236893901KB2021',
    '21R43GH002368939ZZMF2021'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/grant_tool", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        grant_name = request.form.get("grant_name")
        return generate_graph(grant_name)
    return render_template("grant_tool.html", grants=grants)

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

        # If a specific grant is requested, retrieve its data
        grant_status = "Grant not found"
        grant_data = None
        if grant_name:
            grant_data = obligation_progression[obligation_progression['Unique ID'] == grant_name]
            if not grant_data.empty:
                grant_status = f"Grant {grant_name} is {'UDO' if grant_data['UDO Status'].iloc[0] == 'ULO' else 'Non UDO'}"

        # Convert the data to JSON format
        data = {
            "grant_time_elapsed": disbursement_predictions['Grant Time Elapsed'].tolist(),
            "udo_predicted_level": disbursement_predictions['UDO Predicted Level'].tolist(),
            "non_udo_predicted_level": disbursement_predictions['Non UDO Predicted Level'].tolist(),
            "grant_status": grant_status,
            "grant_data": grant_data[['Grant Time Elapsed', 'Obligation Spent']].to_dict(orient='list') if grant_data is not None else None
        }

        return jsonify(data)

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)





