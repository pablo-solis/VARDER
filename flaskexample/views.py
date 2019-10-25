    from flaskexample import app
from flask import render_template
from flask import request
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from utils import *
from utilsVAR import *
from all_utils import *
import re


# the function following the decorator will be fed into
#  an app.route function.....
@app.route('/')
@app.route('/index')
def index():
    # supply dictionary to render template
    index_dict = {'text_1':'Searching for good investments using inflation.'}

    return render_template("index.html",content = index_dict)


@app.route('/VARDERfc')
def VARDERfc():
    # dictionary with all user arguments
    # keys: savings, n_months, year, month, also back_test has value True/False
    u_args = get_u_arguments()

    # expects date, n_steps
    prev,fc_1,first_YOY = generate_forecast_1(u_args['date'],u_args['n_months'])
    strategy = action_from_infl(fc_1)

    #message based on strategy
    message = message_from_strategy(strategy)

    # generate list of examples
    lst = stocks_urls(strategy)
    examples = create_string_of_links(lst)

    # integreate forecast
    # returns a list of YOY values
    fc = integrate_forecast(prev,fc_1,first_YOY)

    # generate YOY_CPI_slice
    # depending on if back_test = True
    # overlap is used when bt=False and says how many previous values to include
    yoy_cpi_df = YOY_CPI_slice(fc,date = u_args['date'],n_steps = u_args['n_months'],bt = u_args['back_test'],overlap = 36)
    infl_df = infl_df_from_CPI(yoy_cpi_df,date = u_args['date'])

    # then generate df of savings with inflation effects alongside
    # a portfolio suggested by VARDER with inflation effects
    vs_df = savings_vs_VARDER_from_YOY_CPI(infl_df,bt = u_args['back_test'],which_tikr=strategy,savings=u_args['savings'])

    # generate a plot from vs_df
    plot_names = ['Money in a Bank','Suggested Investment']

    # all parameters for plot are in this function
    # returns nothing
    generate_plot(vs_df,names = plot_names,user = u_args)
    url = html_plot()



    # 'print' is for trouble shooting
    dict = {'message':message,'print':'','examples':examples}
    return render_template('VARDERfc.html',content = dict,plot_url=url)



@app.route('/VARDER')
def VARDER():
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

    if user_year and user_month:
        #  run a back Test
        df = back_test(year=user_year,month = user_month,n_steps=n_months)


        date = user_year+'-'+user_month+'-'+'01'
        # get comparison, money delta, and tikr name
        versus_df,delta,tikr = bt_tickr_savings_df(date,savings = user_savings,horizon = n_months)
        ax = versus_df.plot()
        ylabel_str = user_month+'-'+user_year+' USD'
        ax.set_ylabel(ylabel_str)
        plt.title('purchasing power of your savings')

        delta_message = interpret_delta(delta)

        bt_url  = html_plot()
        # get suggestion based on size of user_savings
        advice,link = suggestion(user_savings,fund = tikr)

        # generate forecast based on horizon
        fc_values = back_test(n_steps = n_months,today = False).values
        V_dict = {'opportunity':delta_message,'link':link}

        return  render_template('VARDER.html',content = V_dict,plot_url = bt_url)

    else:
        # the prediction for today


        # generate forecast based on horizon
        # currently returns df with just VARDER columns
        cpi_yoy = back_test(n_steps = n_months,today = True)

        fig,loss = purchase_power_VARDER(cpi_yoy,savings = user_savings, horizon = user_horizon)

        op = 'Over '+str(n_months)+' months you can gain '+f'${loss:.2f} in purchasing power.'

        if user_savings>3000:
            action = 'Invest in VIPSX.'
            link = 'https://investor.vanguard.com/mutual-funds/profile/VIPSX'
        else:
            action = 'Use a high yield savings account.'
            link = 'https://www.nerdwallet.com/best/banking/high-yield-online-savings-accounts'

        why = 'message'


        V_dict = {'opportunity':op,'action':action,'link':link}

        return  render_template('VARDER.html',content = V_dict,plot_url =fig)
















@app.route('/sandbox')
def sandbox():
    user_tickr=request.args.get('ticker')
    if user_tickr:
        #get
        page_mode = 'output mode'
        user_table,user_plot,ave_price = return_ticker_table(user_tickr)
        ave_price=  'the average price is '+str(ave_price)
        return render_template('sandbox.html',mode = page_mode,table = user_table, plot_url  = user_plot,ticker = user_tickr,some_info=ave_price)
    else:
        page_mode = 'input mode'
        return render_template('sandbox.html',mode=page_mode)





@app.route('/forecast')
def forecast():
    # get user values
    user_savings = int(request.args.get('savings'))
    user_horizon = request.args.get('horizon')
    n_months =  re.search(r'\d+',user_horizon).group()

    # get suggestion based on size of user_savings
    advice,link = suggestion(user_savings)


    figure,loss = purchase_power(savings = user_savings,horizon = user_horizon)
    fc_dict  = {'text_1':'rising  inflation erodes  the value  of  money','text_2':'here are some ways you can fight inflation', 'text_11':f'over  {n_months} months you can gain  ${loss:.2f} in purchasing power','advice':advice,'link':link}

    return render_template('forecast.html',content = fc_dict,plot_url=figure)
