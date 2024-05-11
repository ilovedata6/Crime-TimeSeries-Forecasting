from flask import Flask, request, jsonify , send_file
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import mplcyberpunk
import base64
import os

app = Flask(__name__)


plt.style.use("cyberpunk")
df = pd.read_csv('./Data/last_8_months.csv')
# df['ps_station'] = df['ps_station'].apply(lambda x: ' '.join(x.split(" ")[1:]))
peak_crime_df = df.copy()
peak_crime_df['datetime'] = pd.to_datetime(peak_crime_df['report_date']+ ' ' +peak_crime_df['report_time'])
peak_crime_df['hour']= peak_crime_df['datetime'].dt.strftime('%I %p')
peak_crime_df['weekofyear'] = peak_crime_df['datetime'].dt.isocalendar().week
peak_crime_df['month'] = peak_crime_df['datetime'].dt.month



@app.route('/forecast_plot', methods=['GET'])
def plot():
    try:
        crime_category = request.args.get('category',default=None)
        police_station = request.args.get('ps_station',default=None)
        if crime_category is None and police_station is None:
            subset_df = df.copy()
        if crime_category is not None and police_station is not None:
            subset_df = df[(df['category']== crime_category) & (df['ps_station']== police_station)]
        elif crime_category is None and police_station is not None:
            subset_df = df[df['ps_station']== police_station]
        elif crime_category is not None and police_station is None:
            subset_df = df[df['category']== crime_category]
        if subset_df is None:
            return jsonify({'error': 'No data found for the given parameters'}), 404
        
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
        model.plot(forecast,xlabel='Date',ylabel="Crime Count",plot_cap=True).savefig("D:/Machine Learning Projects/PSCACrimAnalysis/PSCAFlaskAPI/Graph/Forecast.png")
        
        img_base64 = image_to_base64("D:/Machine Learning Projects/PSCACrimAnalysis/PSCAFlaskAPI/Graph/Forecast.png")
        return jsonify({'image' : img_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:

        img_bytes = img_file.read()

        base64_encoded = base64.b64encode(img_bytes)


        base64_string = base64_encoded.decode('utf-8')

        return base64_string

@app.route('/crimecount', methods=['GET'])
def crimeCount():
    to_date = request.args.get('to_date',default=None)
    from_date = request.args.get('from_date',default=None)

    subset_df = df.copy()
    subset_df = subset_df.groupby('report_date')['report_date'].size()\
                        .reset_index(name="crime")\
                        .sort_values(by="report_date",ascending=True)
    subset_df.rename(columns={'report_date':'ds','crime':'y'},inplace=True)
    model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)

    model.fit(subset_df)

    future_df = model.make_future_dataframe(periods=365, include_history=True)
    forecast = model.predict(future_df)

    try :
        if to_date is not None and from_date is not None:
            forecaseted_range = forecast[(forecast['ds'] <= to_date) & (forecast['ds']>= from_date)]
            if forecaseted_range is not None:
                return jsonify({'Count':{timestamp.strftime('%Y-%m-%d'): round(value) for timestamp, value in forecaseted_range.set_index('ds')['yhat'].items()}})
            else:
                return jsonify({'error': 'No data found for the given Date Range'}), 404
    except IndexError:
        return jsonify({'error': 'No data found for the given date'}), 404

@app.route('/dashboard', methods=['GET'])
def dashboard_stats():
    police_station = request.args.get('police_station',default=None)
    peak_crime_df['report_date'] = pd.to_datetime(peak_crime_df['report_date'])
    
    latest_date = peak_crime_df['report_date'].max()
    top_5_crimes = peak_crime_df['category'].value_counts().nlargest(5).reset_index()
    top_5_crimes_dict = top_5_crimes.to_dict(orient='records')
    
    last_year = latest_date.year
    latest_year_data = peak_crime_df[peak_crime_df['report_date'].dt.year == last_year]
    weekly_data = latest_year_data[latest_year_data['report_date'].dt.isocalendar().week == latest_year_data['report_date'].dt.isocalendar().week.max()]
    category_of_week = weekly_data['category'].mode().iloc[0]
    
    latest_date = peak_crime_df['report_date'].max()
    last_month = latest_date.month
    monthly_df = peak_crime_df[(peak_crime_df['month']==last_month) & (peak_crime_df['year']==last_year)]
    monthly_data = monthly_df[monthly_df['month'] == monthly_df.iloc[0]['month']]
    category_of_month = monthly_data['category'].mode().iloc[0]
    
    if police_station is not None:
        base_path = "./Data/Divisions"
        division = peak_crime_df[peak_crime_df['ps_station']== police_station]['ps_division'].unique()[0]

        folder_path = os.path.join(base_path,division,police_station)
        
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            
            category_count = []
            for file in files :
                ps_df = pd.read_csv(os.path.join(folder_path,file))

                if ps_df.shape[0] > 30 : 

                    ps_df = ps_df.groupby('report_date')['report_date'].size()\
                                .reset_index(name="crime")\
                                .sort_values(by="report_date",ascending=True)
                    ps_df.rename(columns={'report_date':'ds','crime':'y'},inplace=True)
                    model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)

                    model.fit(ps_df)
                    period = 1
                    future_df = model.make_future_dataframe(periods=period, include_history=True)
                    forecast = model.predict(future_df)
                    crimecount = round(forecast.iloc[-1]['yhat'])
                    data = {'category':file[:-4],'count':crimecount}
                    category_count.append(data)

        response = {
            'top_5_crimes': top_5_crimes_dict,
            'category_of_week': category_of_week,
            'category_of_month': category_of_month,
            'Category_count' : category_count
        }
        #
    else : 
        response = {
    'top_5_crimes': top_5_crimes_dict,
    'category_of_week': category_of_week,
    'category_of_month': category_of_month
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5002, debug=True)