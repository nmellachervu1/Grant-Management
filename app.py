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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Langchain AI
from dotenv import load_dotenv
import os
import getpass

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}

# Set the matplotlib backend to 'Agg'
plt.switch_backend('Agg')

app = Flask(__name__)

# List of grants
grants = [
    '20NU2GGH0023029390FKV2021', '22R43GH0023699390JGK2022', '22R43GH0023899390JLL2022', '22R43GH002391CV9390JLL2022',
    '22R43GH002392CV9390JLL2022', '21NU2GGH0023399390FKN2021', '21NU2GGH0023399390HCW2021',
    '21R43GH00236793901A32021', '21R43GH002367939ZZMF2021', '21R43GH00236893901KB2021',
    '21R43GH002368939ZZMF2021', '19NU2GGH00219423HOP9390HEC2022', '21NU2GGH0023789390K872023'
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

@app.route("/")
def index():
    return render_template("index.html")

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
        #country = "SOUTH AFRICA"
        data, grant_data, grant_name2, grant_months_remaining, grantee, grant_obligation, grant_liquidated, grant_udo, udo_percentage, country = generate_graph_with_grant(grant_name)
        #print(country)
        country_area_data, avg_line = generate_country_graph_without_overlay(country)
        #print(avg_line)
        return render_template("grant_chart3.html", data=data, grant_data=grant_data, grant_name = grant_name2, months_remaining = grant_months_remaining, country_area_data = country_area_data, avg_line = avg_line, grantee = grantee, grant_obligation = grant_obligation, grant_liquidated = grant_liquidated, grant_udo = grant_udo, udo_percentage = udo_percentage, country = country)
    return render_template("grant_js_form.html", grants=grants)

@app.route("/global_portfolio")
def portfolio_global():
    #data = generate_graph_without_overlay()
    country = "GLOBAL"
    area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(Global_grants)
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

    return render_template("SA_points6v2.html", data=area_data, latest_months_data=latest_months_data, total_obligations = total_obligations, total_liquidated = total_liquidated, remaining_obligations = remaining_obligations, total_current_UDO=total_current_UDO, UDO_percentage = UDO_percentage, country = "Global", country_area_data = country_area_data, avg_line = avg_line, num_grants = num_grants)

@app.route("/SA")
def portfolio_SA():
    #data = generate_graph_without_overlay()
    country = "SOUTH AFRICA"
    area_data, latest_months_data, total_obligations, total_liquidated, total_current_UDO, UDO_percentage, num_grants = latest_months_in_grants(SA_grants)
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

def ai_summary(docs):
    try: 
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import ChatPromptTemplate

        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Referencing the following: 'You are a graph analyzer assistant. You will be provided a graph that has __ points that represent the latest % Obligation Spent across the % Elapsed Time of grants. These points are different color dots that each represent a unique grant. The red line with a red shaded area below is the area trend of Grants with Less Than %100 Percent Liquidation Pattern, this pattern indicates that these grants are expected to leave behind obligation which requires intervention. If a grant falls within/below this red shaded area, these grants need intervention. The type of intervention depends on the % Time Elapsed of the Grant. If the grant in the red shaded area has less than %50 Grant Time Elapsed, recommend having conversations, reminders and follow ups. If the grant in the red shaded area has more than %50 Grant Time Elapsed, recommend pulling back money and/or amending future obligations. Make sure to mention how the Red Shaded Areas are being calculated. Use a single paragraph format.' Write a concise summary using this data:\\n\\n{context}")]
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
    
def ai_summary_grant(docs):
    try: 
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import ChatPromptTemplate

        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Referencing the following: 'You are a graph analyzer assistant. You will be provided a graph with a line trend represent the latest % Obligation Spent across the % Elapsed Time for a specific grant. The red line with a red shaded area below is the area trend of Grants with Less Than %100 Percent Liquidation Pattern, this pattern indicates that these grants are expected to leave behind obligation which requires intervention. If a grant falls within/below this red shaded area, these grants need intervention.Make sure to mention how the Red Shaded Areas are being calculated using either global or country specific data. Use a single paragraph format.' Write a concise summary using this data:\\n\\n{context}")]
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
        # Load Excel files
        udo_ts = pd.read_excel("GHC FY21-23 Grant UDO Data.xlsx")
        udo_c = pd.read_excel("GHC Grant Data Test.xlsx", skiprows=3, header=1)
        
        # Select relevant columns
        udo_c_selected = udo_c[["Unique ID", "UDO Status", "Recoverable", "Grant Start Date", "Grant End Date", "Grantee", "Country"]]
        
        # Merge dataframes
        udo_combined = udo_ts.merge(udo_c_selected, how='left', on='Unique ID')
        udo_combined.sort_values(by=['Unique ID', 'Month'], axis=0, inplace=True, ignore_index=True)
        
        # Process data
        obligation_progression = udo_combined[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Recoverable", "Grantee", "Country"]]
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
        # Load Excel files
        udo_ts = pd.read_excel("GHC FY21-23 Grant UDO Data.xlsx")
        udo_c = pd.read_excel("GHC Grant Data Test.xlsx", skiprows=3, header=1)
        
        # Select relevant columns
        udo_c_selected = udo_c[["Unique ID", "UDO Status", "Recoverable", "Grant Start Date", "Grant End Date", "Grantee", "Country"]]
        
        # Merge dataframes
        udo_combined = udo_ts.merge(udo_c_selected, how='left', on='Unique ID')
        udo_combined.sort_values(by=['Unique ID', 'Month'], axis=0, inplace=True, ignore_index=True)
        
        # Process data
        obligation_progression = udo_combined[["Unique ID", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Recoverable", "Grantee", "Country"]]
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
        udo_ts = pd.read_excel("GHC FY21-23 Grant UDO Data.xlsx")
        udo_c = pd.read_excel("GHC Grant Data Test.xlsx", skiprows=3, header=1)

        # Select relevant columns
        udo_c_selected = udo_c[["Unique ID","Country","CAN", "Grantee", "Fund Year", "Fund Description",  "UDO Status", "Recoverable", "Grant Start Date", "Grant End Date"]]

        #udo_c_selected = udo_c_selected[udo_c_selected['Grant End Date'] < "2023-09-30"]
        # Merge dataframes
        udo_combined = udo_ts.merge(udo_c_selected, how='left', on='Unique ID')
        udo_combined.sort_values(by=['Unique ID', 'Month'], axis=0, inplace=True, ignore_index=True)

        # Process data
        obligation_progression = udo_combined[["Unique ID", "Country","CAN", "Grantee", "Fund Year", "Fund Description", "Month", "Obligation", "Disbursement", "Undisbursed Amount", "Grant Start Date", "Grant End Date", "UDO Status", "Recoverable"]]

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

        udo_progression = obligation_progression[obligation_progression["UDO Status"] == "ULO"]
        non_udo_progression = obligation_progression[obligation_progression["UDO Status"] == "Non ULO"]

        # Calculate the average line of obligation spent against time elapsed
        avg_obligation_spent = obligation_progression.groupby("Grant Time Elapsed")["Obligation Spent"].mean().reset_index()

        avg_obligation_spent_list = {
            'GrantTimeElapsed': avg_obligation_spent['Grant Time Elapsed'].tolist(),
            'ObligationSpent': avg_obligation_spent['Obligation Spent'].tolist()
        }

        # Apply rolling window to smooth the data
        if Country_Name == 'GLOBAL':
            window = 35
        else:
            window = 20
        
        avg_obligation_spent_list['ObligationSpent'] = pd.Series(avg_obligation_spent_list['ObligationSpent']).rolling(window=window).mean().tolist()

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
    app.run(debug=True)


# In[ ]:




