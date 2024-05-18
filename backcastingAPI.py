from flask import Flask, request, jsonify , send_file
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import mplcyberpunk

app = Flask(__name__)

plt.style.use("cyberpunk")
df = pd.read_csv('./Data/last_8_months.csv')
df['ps_station'] = df['ps_station'].apply(lambda x: ' '.join(x.split(" ")[1:]))
peak_crime_df = df.copy()
peak_crime_df['datetime'] = pd.to_datetime(peak_crime_df['report_date']+ ' ' +peak_crime_df['report_time'])
peak_crime_df['hour']= peak_crime_df['datetime'].dt.strftime('%I %p')
peak_crime_df['weekofyear'] = peak_crime_df['datetime'].dt.isocalendar().week
peak_crime_df['month'] = peak_crime_df['datetime'].dt.month
holidays = pd.read_csv("Data/holidays.csv")

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
        model.plot(forecast,xlabel='Date',ylabel="Crime Count",plot_cap=True).savefig("./PSCAFlaskAPI/Graph/Forecast.png")
        
        return send_file("./PSCAFlaskAPI/Graph/Forecast.png",mimetype="image/png")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    crime_of_week = weekly_data['category'].mode().iloc[0]
    crime_of_week_count = weekly_data['category'].value_counts().iloc[0]


    last_month = latest_date.month
    monthly_df = peak_crime_df[(peak_crime_df['month']==last_month) & (peak_crime_df['year']==last_year)]
    monthly_data = monthly_df[monthly_df['month'] == monthly_df.iloc[0]['month']]
    crime_of_month = monthly_data['category'].mode().iloc[0]
    crime_of_month_count = monthly_data['category'].value_counts().iloc[0]


    latest_year = peak_crime_df['year'].max()
    yearly_df = peak_crime_df[peak_crime_df['year'] == latest_year]
    crime_of_year = yearly_df['category'].mode().iloc[0]
    crime_of_year_count = yearly_df['category'].value_counts().iloc[0]


    if police_station is not None:
        # division = peak_crime_df[peak_crime_df['ps_station']== police_station]['ps_division'].unique()[0]
        Ps_data = peak_crime_df[peak_crime_df['ps_station']== police_station] 
        all_categories = Ps_data['category'].unique()
        category_count = []
        for category in all_categories:
            category_df = Ps_data[Ps_data['category'] == category]
            if category_df.shape[0] > 10 : 

                category_df = category_df.groupby('report_date')['report_date'].size()\
                            .reset_index(name="crime")\
                            .sort_values(by="report_date",ascending=True)
                category_df.rename(columns={'report_date':'ds','crime':'y'},inplace=True)
                model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)

                model.fit(category_df)
                period = 1
                future_df = model.make_future_dataframe(periods=period, include_history=True)
                forecast = model.predict(future_df)
                crimecount = round(forecast.iloc[-1]['yhat'])
                data = {'category':category,'count':crimecount}
                category_count.append(data)

        response = {
            'top_5_crimes': top_5_crimes_dict,
            'crime_of_week': crime_of_week,
            'crime_of_week_count': str(crime_of_week_count),
            'crime_of_month': crime_of_month,
            'crime_of_month_count': str(crime_of_month_count),
            'crime_of_year' : crime_of_year,
            'crime_of_year_count' : str(crime_of_year_count),
            'category_count' : category_count
        }
        
    else : 
        response = {
    'top_5_crimes': top_5_crimes_dict,
    'crime_of_week': crime_of_week,
    'crime_of_week_count': str(crime_of_week_count),
    'crime_of_month': crime_of_month,
    'crime_of_month_count': str(crime_of_month_count),
    'crime_of_year' : crime_of_year,
    'crime_of_year_count' : str(crime_of_year_count)
    }

    return jsonify(response)

# Overall crimecount on hourly basis 

@app.route("/peak_hour")
def peak_hour():
    crime_category = request.args.get('crime_category',default=None)
    crime_category = crime_category.upper()
    peak_crime_df = df.copy()
    peak_crime_df['datetime'] = pd.to_datetime(peak_crime_df['report_date']+ ' ' +peak_crime_df['report_time'])
    peak_crime_df['hour']= peak_crime_df['datetime'].dt.strftime('%I %p')
    if crime_category is not None:
        peak_crime_df = peak_crime_df[peak_crime_df['category'] == crime_category]

        if not peak_crime_df.empty:
            peak_crime_groupby = peak_crime_df.groupby('hour')['hour'].count().reset_index(name="count")
            peak_crime_groupby['hour'] = peak_crime_groupby['hour'].astype(str)
            return jsonify({'peak_hour': peak_crime_groupby.to_dict(orient='records')})
        else:
            return jsonify({'error': 'No data found for the given crime category'}), 404
    else:
        return jsonify({'error': 'No Crime Category Found'}), 404


# Top 5 Crimewise Police Stations on OverAll data

@app.route("/ps_crime")
def ps_crime():
    top_psCrime = df.copy()
    top_psCrime_groupby = top_psCrime.groupby('ps_station')['ps_station']\
                                  .count().reset_index(name="crime")\
                                  .sort_values(by="crime",ascending=False)
                                    
    total_crime_count = top_psCrime_groupby['crime'].sum()
    top_psCrime_groupby['percentage'] = round((top_psCrime_groupby['crime'] / total_crime_count) * 100,2)
    top_psCrime_groupby = top_psCrime_groupby.head(5)
    top_psCrime_groupby = top_psCrime_groupby.set_index('ps_station')[['crime','percentage']].to_dict()


    return jsonify(top_psCrime_groupby)


@app.route("/backcasting")
def backcasting():
    event_name = request.args.get('event_name',default=None)
    police_station = request.args.get('police_station',default=None)
    analysis_days = int(request.args.get('analysis_days',default=None))
    other_date = request.args.get('other_date',default=None)
    backcast_df = df.copy()
    holidays_df = holidays.copy()
    backcast_df['report_date'] = pd.to_datetime(backcast_df['report_date'])
    holidays_df['Name'] = holidays_df.apply(lambda x: x['Name'].lower(),axis=1)
    if event_name is not None:
        if holidays_df['Name'].str.contains(event_name, case=False).any():

            event_date = holidays_df.loc[holidays_df['Name'] == event_name.lower(), 'Date'].iloc[0]
            before_event_df = backcast_df[(backcast_df['report_date'] >= pd.to_datetime(event_date) - pd.Timedelta(days=analysis_days)) \
                                & (backcast_df['report_date'] <= pd.to_datetime(event_date))]
            before_ps_df = before_event_df[before_event_df['ps_station'] == police_station]

            if not before_ps_df.empty:
                before_event_dict = before_ps_df['category'].value_counts().to_dict()
            
            after_event_df = backcast_df[(backcast_df['report_date'] >= pd.to_datetime(event_date)) \
                                & (backcast_df['report_date'] <= pd.to_datetime(event_date) + pd.Timedelta(days=analysis_days))]

            after_ps_df = after_event_df[after_event_df['ps_station'] == police_station]

            if not after_ps_df.empty:
                after_event_dict = after_ps_df['category'].value_counts().to_dict()
    else:
        before_event_df = backcast_df[(backcast_df['report_date'] >= pd.to_datetime(other_date) - pd.Timedelta(days=analysis_days)) \
                            & (backcast_df['report_date'] <= pd.to_datetime(other_date))]
        before_ps_df = before_event_df[before_event_df['ps_station'] == police_station]
        if not before_ps_df.empty:
            before_event_dict = before_ps_df['category'].value_counts().to_dict()

        after_event_df = backcast_df[(backcast_df['report_date'] >= pd.to_datetime(other_date)) \
                            & (backcast_df['report_date'] <= pd.to_datetime(other_date) + pd.Timedelta(days=analysis_days))]

        after_ps_df = after_event_df[after_event_df['ps_station'] == police_station]

        if not after_ps_df.empty:
            after_event_dict = after_ps_df['category'].value_counts().to_dict()

    response = {}
    for crime in set(before_event_dict.keys()).union(after_event_dict.keys()):
            response[crime] = {
                    'before': before_event_dict.get(crime, 0),
                    'after': after_event_dict.get(crime, 0)
}
    return jsonify(response)
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5002, debug=True)