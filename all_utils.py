from flask import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import yfinance as yf
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
import base64
import re

# Import Statsmodels
from statsmodels.tsa.api import VAR

import  seaborn as  sns
sns.set_style('darkgrid')


#----- handling arguments
# intended usage:
# should return dictionary
# u_args = get_u_arguments()
# u_args['savings'] = user_savings etc

def get_u_arguments():
    # get user values
    user_savings = int(request.args.get('savings'))
    user_horizon = request.args.get('horizon')
    try:
        n_months =  int(re.search(r'\d+',user_horizon).group())
    except:
        n_months = 6

    # check if bt vars are set
    user_year = request.args.get('bt_year')
    user_month =request.args.get('bt_month')

    # if the above are None then set to the current date
    if user_year and user_month:
        back_test = True
    else:
        user_year = '2019'
        user_month = '08'
        back_test = False

    u_args = {'savings':user_savings,
                'n_months':n_months,
                'year':user_year,
                'month':user_month,
                'date':user_year+'-'+user_month+'-'+'01',
                'back_test': back_test}
    return u_args



#--- loading formatting data

# main data I'm working with
# last data point I have is for 2019-08
def load_sentiment_YOY_CPI():
    df = pd.read_csv('flaskexample/static/neg_CPI_YOY.csv',index_col='Unnamed: 0',parse_dates=True)
    return df[['neg/l','YOY','CPI']]



def crop_data(df,date,n_steps):
    # note when date is most recent, e.g. 2019-08-01
    # then test is empty

    # date should be string or pandas datetime
    # n is number of additonal rows to get

    # get i loc of date
    i = df.index.get_loc(date)
    # return everything up to i plus the next n values
    #  before 12 the values are nans
    train = df.iloc[12:(i+1)]
    test = df.iloc[(i+1):(i+1)+n_steps]
    return train,test


def load_VWELX():
    df = pd.read_csv('flaskexample/static/VWELX.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df

def load_VIPSX():
    df = pd.read_csv('flaskexample/static/VIPSX.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df


def load_CPI():
    df = pd.read_csv('flaskexample/static/CPI.csv',index_col = 'Unnamed: 0',parse_dates=True)
    return df




#---- generate forecast

def integrate_forecast(prev,fc,first):
    # returns a list
    # previous values,fc to append, first value of undifferenced forecase
    # append
    arr = np.append(prev,fc)
    arr = np.append(first,arr)
    result = []
    for i in range(0,len(arr)):
        result.append(sum(arr[:i+1]))
    return result


# how to interpret the inflation prediction
# this about a third option: treasury bonds
def action_from_infl(fc_values):
    signal = np.mean(fc_values)
    # i might want to switch this...
    if signal>0.005:
        # rising inflation
        return 'TIPS'
    elif signal<-0.005:
        # deflation
        return 'Treasury Bonds'
    else:
        return 'S&B'
def message_from_strategy(strategy):
    # what's happening with inflation
    temp_1 = 'Infaltion is predicted to '
    temp_2 = 'VARDER recommends investing in '
    temp_3 = ' Here are some examples:'
    dd = {'TIPS':temp_1+'rise. '+temp_2+'inflation protected securities. '+temp_3,
            'S&B':temp_1+'be relatively tame. '+temp_2+'a stock and bonds portfolio.'+temp_3,
            'Treasury Bonds':temp_1+'fall. '+temp_2+'Treasury bonds.'+temp_3}
    return dd[strategy]

# list of stocks based on strategy
def stocks_urls(strategy):
    SnB_list = [('VTI','https://finance.yahoo.com/quote/VTI'),('BND','https://finance.yahoo.com/quote/BND'),('FXAIX','https://finance.yahoo.com/quote/FXAIX?p=FXAIX&.tsrc=fin-srch'),('VWELX','https://finance.yahoo.com/quote/VWELX')]
    TIPS_list = [('SCHP','https://finance.yahoo.com/quote/SCHP?p=SCHP&.tsrc=fin-srch'),('VIPSX','https://finance.yahoo.com/quote/VIPSX?p=VIPSX&.tsrc=fin-srch'),('FIPDX','https://finance.yahoo.com/quote/FIPDX?p=FIPDX&.tsrc=fin-srch'),('TDTT','https://finance.yahoo.com/quote/TDTT?p=TDTT&.tsrc=fin-srch'),('STIP',('https://finance.yahoo.com/quote/STIP?p=STIP&.tsrc=fin-srch'))]
    T_Bond_list = [('DTYL','https://finance.yahoo.com/quote/DTYL?p=DTYL&.tsrc=fin-srch'),('DTUS','https://finance.yahoo.com/quote/DTUS?p=DTUS&.tsrc=fin-srch'),('DTYS','https://finance.yahoo.com/quote/DTYS?p=DTYS&.tsrc=fin-srch'),('EDV','https://finance.yahoo.com/quote/EDV?p=EDV&.tsrc=fin-srch')]

    dd = {'TIPS':TIPS_list,'S&B':SnB_list,'Treasury Bonds':T_Bond_list}
    return dd[strategy]

def create_string_of_links(lst):
    # list should be pairs (TIKR,url)
    links = [f'<a href="{url}">{TIKR}</a>' for TIKR,url in lst]
    string = ', '.join(links)
    return string

# year = 2019 and month = 08, later that will be 09
# note the _1 means we are looking at difference
def generate_forecast_1(date='2003-01-01',n_steps = 6):
    # n_steps is how far into future you look
    # crop the data depending on n_steps and date
    neg_YOY_CPI = load_sentiment_YOY_CPI()

    # if date is most recent then test is empty
    train, test=crop_data(neg_YOY_CPI,date,n_steps)

    #take first difference and record first row
    first_row = train.iloc[0]
    train_1 = train.diff().dropna()
    first_YOY = first_row['YOY']
    prev = train_1.values[:,1]

    model = VAR(train_1,freq = 'MS') # create VAR model
    results = model.fit(4)  #for now fit to 4
    lag_order = results.k_ar
    prediction_input = train_1.values[-lag_order:]

    # I want last column
    infl_results = results.forecast(prediction_input, n_steps)[:,1]

    # return triple: previous, forecast_1, first_YOY

    return prev,infl_results,first_YOY


# generate CPI YOY slice
def YOY_CPI_slice(fc,date = '2003-01-01',n_steps = 6,bt = True,overlap = 36):
    # fc is integrated forecast of YOY forecast
    # set start and end dates
    start = pd.to_datetime(date)
    end_date = start+pd.offsets.MonthBegin(n_steps)
    all_YOY_CPI = load_sentiment_YOY_CPI()

    # return slice when bt is True
    if bt:
        # just return historical slice
        # no need to utilize fc
        return all_YOY_CPI[['YOY','CPI']][start:end_date]
    else:
        # generate the CPI from YOY forecast
        # here we use keyword arg overlap
        idx = pd.date_range(end = end_date,freq = 'MS',periods = n_steps+overlap)
        values = fc[-(n_steps+overlap):]
        fc_df = pd.DataFrame({'YOY':values})
        fc_df.index = idx
        # now need to add CPI data based on YOY
        YOY_CPI_fc = all_YOY_CPI[['YOY','CPI']].reindex(index = fc_df.index)


        # now update the YOY values
        YOY_CPI_fc['YOY'] = fc_df['YOY']

        # now compute 'CPI' from 'YOY'
        YOY_CPI_fc['CPI'] = ((YOY_CPI_fc['YOY']/100)+1)*YOY_CPI_fc['CPI'].shift(12)

        return YOY_CPI_fc



#---- purchasing power

# after using this will not need YOY values or CPI values
def infl_df_from_CPI(YOY_CPI_df,date = '2003-03-01'):
    base = YOY_CPI_df['CPI'][date]
    # create series
    infl_series = YOY_CPI_df['CPI']/base -1
    new_df = pd.DataFrame({'infl':infl_series.values})
    new_df.index = YOY_CPI_df.index
    # note first 12 values will be nans
    return new_df

def savings_vs_VARDER_from_YOY_CPI(infl_df,bt = True, which_tikr="TIPS",savings=3000):
    # infl_df has 'infl' column
    # add effect of inflation to savings over time
    infl_df['Money in a Bank'] = savings/(1+infl_df['infl']) # infaltion decay
    if bt:
        # if doing a back test VARDER col is based on historical values
        tikr_monthly = load_VWELX() if which_tikr == 'S&B' else load_VIPSX()
        start = infl_df.index[0]
        end = infl_df.index[-1]
        tikr_monthly_slice = tikr_monthly[start:end]

        # do some basic manipulation
        first_price = tikr_monthly_slice['Close'].values[0]
        n_shares = int(savings/first_price)
        remainder = savings - n_shares*first_price

        # multiply by number of shares + remainder
        tikr_monthly_slice['invested'] = tikr_monthly_slice['Close']*n_shares+remainder

        # add effects of inflation
        # could make name customizable by passing which_tikr
        # for now have a static name
        infl_df['Suggested Investment'] = tikr_monthly_slice['invested']/(1+infl_df['infl'])
        return infl_df


    else:
        # generate values based on expected returns
        apy = 7.10
        mpy =  (1+apy/100)**(1/12)

        # find index location of base
        inv_start = infl_df.index.get_loc('2019-08-01')
        old_list = [savings]*len(infl_df)

        n_inv = len(old_list[inv_start:]) # change this many values

        inv_values = [savings*(mpy)**(i) for i in range(1,n_inv+1)]
        # old_list[-(len(inv_values)):] = inv_values
        new_list = old_list[:inv_start]+inv_values

        # have static name for now to be compatible with backtest
        infl_df['Suggested Investment'] = new_list/(1+infl_df['infl'])
        return infl_df

#---- generate plot
def html_plot():
    # ----- save image as a crazy string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    base64.b64encode(img.getvalue())

    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_plot(df,names=['Money in a Bank','Suggested Investment'],user = {'date':'2003-01-01'}):
    # df has inflation column, only plot the cols from plot names
    # drop any non values (they appear when not doing backtest)
    plot_df = df[names].dropna()

    # get the values to plot manually
    x = plot_df.index.values
    y0 = plot_df[names[0]].values
    y1 = plot_df[names[1]].values

    # now get the last values to get a vertical line
    last_x = x[-1]
    ly0 = y0[-1]
    ly1 = y1[-1]

    # make the plot
    fig,ax = plt.subplots()
    # plot y1 first to make plot clearer
    ax.plot(x,y1,'c') # suggested investment is below, cyan
    ax.plot(x,y0,'b') # Money in bank is on top, blue
    # ax.plot([last_x, last_x],[ly0,ly1],'g--', lw=2)  # vertical line
    ax.vlines(last_x,ly0,ly1,colors='g',linestyles='dashed')  # vertcial line
    ax.set_title('Opportunity Gain of Investing',fontsize=25)
    ax.set_ylabel('Purchasing Power in '+user['date']+' USD')
    plt.xticks(rotation = 25)
    num = abs(ly0-ly1)
    vlabel = f'${num:.2f}'
    # switch order of names
    ax.legend([names[1],names[0],vlabel])

    # set the ylim so the scale is not misleading
    # find min and max to set things appropriately
    y_min_value = min(y1)
    y_max_value = max(y1)
    ymin = int(0.93*y_min_value)
    ymax = int(1.07*y_max_value)
    ax.set_ylim([ymin,ymax])

    #x lower limit
    # users starting date minus a few months etc
    x_lower = pd.to_datetime(user['date'])+pd.offsets.MonthBegin(-4)
    # only if back_test is false
    if user['back_test']:
        pass # for back test the lower limit gives a bad plot
    else:
        ax.set_xlim(left = x_lower) # set a lower limit



#---------
