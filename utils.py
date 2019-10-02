import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import yfinance as yf
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
import base64
import re
import  seaborn as  sns

sns.set_style('darkgrid')

'''
sia = SentimentIntensityAnalyzer()
def score_from_txt(txt, col='compound'):
    return sia.polarity_scores(txt)[col]
'''

def suggestion(savings, fund = 'VIPSX'):
    if savings<3000:
        advice = 'Consider transferring your money into a high interest savings account.'
        link ='https://www.nerdwallet.com/best/banking/high-yield-online-savings-accounts'
    else:
        advice =  f'Consider investing in {fund}.'
        link = 'https://investor.vanguard.com/mutual-funds/profile/VIPSX' if fund == 'VIPSX' else 'https://investor.vanguard.com/mutual-funds/profile/VWELX'
    return advice,link


def return_ticker_table(tickr):
    data = yf.download(tickr,'1988-01-01','2018-01-01')
    data.Close.plot()
    figure = html_plot()
    ave_price  = data.Close.mean()
    # the close prices
    return data[['Close']].head(10).append(data[['Close']].tail(10)).to_html(), figure, ave_price
