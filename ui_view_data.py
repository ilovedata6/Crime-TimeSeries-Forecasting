import pandas as pd
import matplotlib.pyplot as plt
# from prophet.serialize import model_to_json, model_from_json
from colorama import Fore
import datetime
import mplcyberpunk
from prophet import Prophet
import streamlit as st
import pandas as pd
from datetime import datetime


def get_data(file_path):
    """Loads the dataset from the specified Excel file path."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("Error: CSV file not found.")
        return None

def display_dataset(df):
    """Displays the dataset as an interactive table with sorting and filtering capabilities."""
    st.dataframe(df)  # Improve formatting

def display_analysis_graph(predictive_df):
    subset_df = predictive_df[predictive_df['report_date'] >= '2023-07-01']
    subset_df['report_date'] = pd.to_datetime(subset_df['report_date'])
    subset_df = subset_df.groupby('report_date')['report_date'].size()\
                        .reset_index(name="crime")\
                        .sort_values(by="report_date",ascending=True)

    subset_df.rename(columns={'report_date':'ds','crime':'y'},inplace=True)
    model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)

    model.fit(subset_df)
    period = 7
    future_df = model.make_future_dataframe(periods=period, include_history=True)
    forecast = model.predict(future_df)
    fig = model.plot(forecast)         #.savefig("forecasted.png")
    # plt.xlabel('Date',color='black')
    # plt.ylabel("Crime",color='black')
    # plt.xticks(color='black')
    # plt.yticks(color='black')
    # plt.show()
    
    fig.set_label("Forecast")

    st.pyplot(fig)

def get_formatted_date(date_obj):
    """Formats the selected date from the calendar picker in YYYY-MM-DD format."""
    return date_obj.strftime("%Y-%m-%d")

def main():
    """Streamlit app to display dataset, generate analysis graph, and handle date selection."""
    st.title("Predictive Crime Analysis App")

    # Upload Excel file
    uploaded_file = st.file_uploader("Upload Excel File", type="csv")
    if uploaded_file is not None:
        df = get_data(uploaded_file)

        if df is not None:
            st.subheader("Lahore Crime Data")
            display_dataset(df)

            # Analysis and date selection section
            col1, col2 = st.columns(2)

            # Analysis button and graph display
            with col1:
                if st.button("Analyse"):
                    predictive_df = df.copy()

                    display_analysis_graph(predictive_df)  # Avoid modifying original df

            # Calendar picker and date-based graph display
            with col2:
                predictive_df = df.copy()
                subset_df = predictive_df[predictive_df['report_date'] >= '2023-07-01']
                subset_df['report_date'] = pd.to_datetime(subset_df['report_date'])
                subset_df = subset_df.groupby('report_date')['report_date'].size()\
                                    .reset_index(name="crime")\
                                    .sort_values(by="report_date",ascending=True)

                subset_df.rename(columns={'report_date':'ds','crime':'y'},inplace=True)
                selected_date = st.date_input("Select Date (YYYY-MM-DD)")
                formatted_date = get_formatted_date(selected_date)
                model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)

                model.fit(subset_df)
                period = 365
                future_df = model.make_future_dataframe(periods=period, include_history=True)
                forecast = model.predict(future_df)
                try :
                    predicted_count = forecast[forecast['ds'] == formatted_date]['yhat'].values[0]
                    st.write(f"Predicted Number of Crimes on {formatted_date}: {round(predicted_count)}")
                except IndexError:
                    st.error("Choose a valid date within 1 year")
    else :
        st.info("Please upload an Actual Excel file.")
if __name__ == "__main__":
    main()
