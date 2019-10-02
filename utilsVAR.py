import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
import base64
import re
import seaborn as  sns
import bls

# Import Statsmodels
from statsmodels.tsa.api import VAR

from random import choice

sns.set_style('darkgrid')

# -------- for plotting  -----------
def html_plot():
    # ----- save image as a crazy string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    base64.b64encode(img.getvalue())

    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url
#---------


# generates the main plot for current prediction
def purchase_power_VARDER(df,savings=1000,horizon='3months'):
    # get number of months
    x  = re.search(r'\d+',horizon)
    n_months = int(x.group())

    # df should have columns CPI,YOY
    # generate purchase power plot based on model
    base_pp = savings
    base = df.loc['2019-08-01','CPI']
    df['infl'] = (df['CPI']/base - 1) #  definition of inflation
    df['purchase_power'] = savings/(1+df['infl']) # infaltion decay

    # create VARDER forcast
    # add behavior with investing
    # assume a  7.10 growth rate
    # also account if savings is less than 3000
    if savings>3000:
        apy = 7.10
        mpy =  (1+apy/100)**(1/12)
    else:
        apy = 2.5
        mpy =  (1+apy/100)**(1/12)

    # find index location of base
    inv_start = df.index.get_loc('2019-08-01')
    old_list = [savings]*len(df)

    # change this many values
    n_inv = len(old_list[inv_start:])

    inv_values = [base_pp*(mpy)**(i) for i in range(1,n_inv+1)]
    old_list[-(len(inv_values)):] = inv_values
    new_list = old_list[:inv_start]+inv_values

    new_df = pd.DataFrame({'purchase_power':df['purchase_power'].values,'VARDER':old_list})
    new_df.index = df.index

    # add inflation effects
    new_df['VARDER']=new_df['VARDER']/(1+df['infl'])

    VARDER_return = new_df['VARDER'].values[-1]


    # see  the decay over their horizon
    # use inv_start
    #  get value  of purchase_power  at  i+n_months
    temp= df['purchase_power'].iloc[inv_start+n_months]

    # get the last value of df['VARDER']
    loss= VARDER_return  - temp #  should be positive number


    #  create a  nice plot
    ax  = new_df[['purchase_power','VARDER']].tail(n_months+3).plot()
    ax.set_ylabel('Aug 2019 usd')
    plt.title('purchasing power of your savings')

    # store as 64bitstring
    figure  = html_plot()
    return figure, loss

def historical_VWELX():
    df = pd.read_csv('flaskexample/static/VWELX.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df

def historical_VIPSX():
    df = pd.read_csv('flaskexample/static/VIPSX.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df


def historical_CPI():
    df = pd.read_csv('flaskexample/static/CPI.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df


def bt_tickr_savings_df(date = '2017-08-01',savings=1000,horizon=3):
    # this is where the model takes ACTION
    s_year = date[:4]
    s_month = date[5:7]


    tikr = signal_from_infl(year = s_year,month = s_month,n_steps = horizon)
    # tikr = choice(['VIPSX','VWELX'])

    tikr_monthly = historical_VWELX() if tikr == 'VWELX' else historical_VIPSX()

    # get the slice based on dates and horizon
    start = pd.to_datetime(date)
    end = start+pd.offsets.MonthBegin(horizon)
    tikr_monthly_slice = tikr_monthly[start:end]

    first_price = tikr_monthly_slice['Close'].values[0]
    n_shares = int(savings/first_price)
    remainder = savings - n_shares*first_price

    # multiply by number of shares + remainder
    tikr_monthly_slice['invested'] = tikr_monthly_slice['Close']*n_shares+remainder

    # get cpi data use to compute inflation
    cpi_df = historical_CPI()
    cpi_trim = cpi_df[start:end]

    new_df = pd.DataFrame({tikr:tikr_monthly_slice['invested'].values,'CPI':cpi_trim['CPI'].values})

    new_df.index = tikr_monthly_slice.index

    # set base to calculate inflation
    base = new_df['CPI'].values[0]
    new_df['infl'] = new_df['CPI']/base -1
    new_df['Savings'] = savings/(1+new_df['infl'])
    new_df[tikr] = new_df[tikr]/(1+new_df['infl'])

    delta = new_df[tikr].values[-1] - new_df['Savings'].values[-1]

    return new_df[[tikr,'Savings']],delta,tikr

def interpret_delta(delta):
    if delta>0:
        return f'Investing leads to a gain of ${delta:.2f} in purchasing power.'
    else:
        return f'Investing leads to a loss of ${abs(delta):.2f} in purchasing power.'

def bt_purchase_power(df,tikr='VWELX',date = '2017-08-01',savings=1000,horizon='3months'):
    #
    # df should have columns CPI and YOY
    # but for now I'll just need CPI
    # make sure to provide date

    # get number of months
    x  = re.search(r'\d+',horizon)
    n_months = int(x.group())

    # df should have columns CPI,YOY
    # generate purchase power plot based on model
    base_pp = savings
    base = df.loc[date,'CPI']
    df['infl'] = (df['CPI']/base - 1) #  definition of inflation
    df['purchase_power'] = savings/(1+df['infl']) # infaltion decay

    # now all I need to do is get actual stock data

def infl_forecast_values(year='2001',month='02',n_steps = 6):
    # n_steps is how far into future you look
    # crop the data depending on n_steps and date
    orig_df = load_data()
    date = form_date(year,month)
    train, test=crop_data(orig_df,date,n_steps)

    #take first difference
    first_row, train_1 =  take_diff(train)
    first_YOY = first_row['YOY']

    # create VAR model
    model = VAR(train_1,freq = 'MS')

    #for now fit to 4
    results = model.fit(4)


    lag_order = results.k_ar
    prediction_input = train_1.values[-lag_order:]

    # I want last column
    infl_results = results.forecast(prediction_input, n_steps)[:,1]
    return infl_results

def signal_from_infl(year='2001',month='02',n_steps = 6):
    # get forecast
    fc = infl_forecast_values(year='2001',month='02',n_steps = 6)
    signal = sum(fc)

    if signal>0:
        return 'VWELX'
    else:
        return 'VIPSX'


def back_test(year='2001',month='02',n_steps = 6,today = False):
    # n_steps is how far into future you look
    # if today then don't need to compute mean square error
    if today:
        date = form_date('2019','08')
    else:
        date = form_date(year,month)

    # used when today = True
    end_date= pd.to_datetime(date)+pd.offsets.MonthBegin(n_steps)
    orig_df  = load_data()

    # crop the data depending on n_steps and date
    train, test=crop_data(orig_df,date,n_steps)

    #take first difference
    first_row, train_1 =  take_diff(train)
    first_YOY = first_row['YOY']

    # create VAR model
    model = VAR(train_1,freq = 'MS')

    #for now fit to 4
    results = model.fit(4)


    lag_order = results.k_ar
    prediction_input = train_1.values[-lag_order:]

    # I want last column
    infl_results = results.forecast(prediction_input, n_steps)[:,1]

    #previous inflation values
    prev_infl = train_1.values[:,1]

    # integrate fc
    infl_with_fc = integrate_prediction(prev_infl,infl_results,first_YOY)

    # just return results for current
    if today:
        # returns mean lower upper
        # fig = results.plot_forecast(10)
        # results.forecast_interval(y = prediction_input,steps = 6)

        # create prediction index
        # could change overlap later
        # want overlap so that I can compute CPI from YOY
        overlap = 24
        idx = pd.date_range(end = end_date,freq = 'MS',periods = n_steps+overlap)
        values = infl_with_fc[-(n_steps+overlap):]
        fc_df = pd.DataFrame({'YOY':values})
        fc_df.index = idx

        # now need to add CPI data based on YOY
        YOY_CPI_fc = orig_df[['YOY','CPI']].reindex(index = fc_df.index)

        # now update the YOY values
        YOY_CPI_fc['YOY'] = fc_df['YOY']
        # now compute 'CPI' from 'YOY'
        YOY_CPI_fc['CPI'] = ((YOY_CPI_fc['YOY']/100)+1)*YOY_CPI_fc['CPI'].shift(12)

        m = len(YOY_CPI_fc)
        # return the non nan values
        YOY_CPI_fc.tail(m-12)

        return YOY_CPI_fc

    # create dataframe with prediction...
    # should return orig with VARDER column series for YOY fc
    with_fc_df = append_fc(orig_df,date,n_steps,fc=infl_with_fc)

    # now you need to integrate results
    return with_fc_df

#  for start of back test
def form_date(year,month):
    return year+'-'+month+'-'+'01'
# main data I'm working with
def load_data():
    df = pd.read_csv('flaskexample/static/neg_CPI_YOY.csv',index_col='Unnamed: 0',parse_dates=True)
    return df[['neg/l','YOY','CPI']]
form_date('1997','02')

# crop data accorind to date
def crop_data(df,date,n):
    # date should be string or pandas datetime
    # n is number of additonal rows to get

    # get i loc of date
    i = df.index.get_loc(date)
    # return everything up to i plus the next n values
    #  before 12 the values are nans
    train = df.iloc[12:(i+1)]
    test = df.iloc[(i+1):(i+1)+n]
    return train,test
def append_fc(df,date,n,fc=0):
    # date should be string or pandas datetime
    # n is number of additonal rows to get

    # get i loc of date
    i = df.index.get_loc(date)
    # return everything up to i plus the next n values
    df = df.iloc[12:(i+1)+n]

    # now df and fc should have same length
    df['VARDER'] = fc
    return df

def append_full_df():
    # return all columns

    pass

def take_diff(df):
    #  record first row
    first_row = df.iloc[0]
    return first_row,  df.diff().dropna()

# just takes list and value
def arr_undo_diff(lst,val):
    temp = [val]+lst
    result = []
    for i in range(0,len(temp)):
        result.append(sum(temp[:i+1]))
    return result

def integrate_prediction(prev,fc,first):
    # previous values,fc to append, first value of undifferenced forecase
    # append
    arr = np.append(prev,fc)
    arr = np.append(first,arr)
    result = []
    for i in range(0,len(arr)):
        result.append(sum(arr[:i+1]))
    return result
