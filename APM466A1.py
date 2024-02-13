import numpy as np
import pandas as pd
import datetime
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from matplotlib import pyplot as plt
from dateutil.relativedelta import relativedelta
import os
print(os.path.abspath('.'))
      
def flow(today,I_date,M_date,Coupon_rate):
    dates=[]
    date_0=M_date
    while date_0>today:
        dates.insert(0,date_0)
        date_0=date_0-relativedelta(months=6)
    if I_date>date_0:
        date_0=I_date
    r_days=(today-date_0).days
    next_date=dates[0]
    b_days=(next_date - date_0).days
    r_interest=r_days/b_days*Coupon_rate
    days=np.array([(date-today).days for date in dates])
    years=days/365
    cashflow=np.ones_like(years)*Coupon_rate
    cashflow[-1]+=100
    return days,years,cashflow,r_interest


def bcf(bond,today,df_bonds,df_prices):
    price=df_prices[df_prices['Date']==today][bond].item()
    temp=df_bonds[df_bonds['ISIN']==bond]
    Coupon_rate=temp['Coupon_rate'].item()
    I_date=temp['I_date'].item()
    M_date=temp['M_date'].item()
    day,time,cashflow,accrued=flow(today,I_date,M_date,Coupon_rate)
    dirty=accrued+price
    return time, cashflow, dirty

def d(r,time,cashflow,dirty):
    return dirty-sum(cashflow*np.exp(-r*time))

dtype={'coupon':float,'ISIN':str}
df_bonds=pd.read_csv('bonds.csv',parse_dates=['M_date','I_date'],usecols=['Coupon_rate','M_date','I_date','ISIN'])
df_prices=pd.read_csv('Prs.csv',parse_dates=['Date'], )
bonds=['CA135087J967', 'CA135087K528', 'CA135087K940', 'CA135087L518', 'CA135087L930', 'CA135087M847', 'CA135087N837','CA135087P576', 'CA135087Q491', 'CA135087Q988']
dates=df_prices['Date'].to_list()

def ytm(dates, bonds, df_bonds, df_prices):
    df_ytm=[]
    for today in dates:
        term=[]
        ytm=[]
        for bond in bonds:
            temp=df_bonds[df_bonds['ISIN'] == bond]
            M_date=temp['M_date'].item()
            term.append((M_date-today).days) 
            bond_inf=bcf(bond, today, df_bonds, df_prices)
            Res=root(d, 0.01, args=bond_inf)
            ytm.append(Res['x'].item())
        today_s=datetime.datetime.strftime(today, '%b %d, %Y')
        df_ytm.append({'date': today_s, 'rate': ytm, 'term': term})
    return df_ytm

df_ytm=ytm(dates,bonds,df_bonds,df_prices)
fig, ax=plt.subplots()
for curve in df_ytm:
    ax.plot(curve['term'], curve['rate'], label=curve['date'])
plt.legend()
ax.set_title('yield curve')
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels(range(6))
ax.set_xlabel('term')
ax.set_ylabel('YTM')
plt.show()

def time_rate(time, i_time, i_rate):
    if time>i_time[-1]:
        return False
    else:
        for i in range(len(i_time)):
            if time>i_time[i]:
                continue
            elif time==i_time[i]:
                return i_rate[i]
            else:
                I_rate = (1-(time-i_time[i-1])/(i_time[i]-i_time[i-1]))*i_rate[i]+(time-i_time[i-1])/(i_time[i]-i_time[i-1])*i_rate[i-1]
                return I_rate

def pr(r, *args):
    args = list(*args)
    p = args.pop(0)
    for i in range(len(args)//2):
        f1=args[2*i]
        f2=args[2*i+1]
        p=p-f1*f2**r
    return p

def spot(dates, bonds, df_bonds, df_prices):
    df_spot = []
    for today in dates:
        i_time = []
        i_rate = []
        term = []
        for bond in bonds:
            temp = df_bonds[df_bonds['ISIN'] == bond]
            M_date = temp['M_date'].item()
            term.append((M_date - today).days)
            time, cashflow, price = bcf(bond, today, df_bonds, df_prices)
            if not i_time:
                R = np.log(cashflow / price) / time
                i_time.append(time[0])
                i_rate.append(R[0])
            else:
                args = []
                t_new = time[-1]
                t_last = i_time[-1]
                r_last = i_rate[-1]
                for i in range(len(time)):
                    R = time_rate(time[i], i_time, i_rate)
                    if R:
                        dcf = cashflow[i]*np.exp(-R* time[i])
                        price = price - dcf
                    else:
                        if t_last<time[i]<=t_new:
                            prop_last = (t_new - time[i])/(t_new - t_last)
                            prop_new = 1- prop_last
                            f1 = cashflow[i]*np.exp(-prop_last*r_last*time[i])
                            f2 = np.exp(-prop_new*time[i])
                            args.append(f1)
                            args.append(f2)
                args.insert( 0, price)
                Res=root(pr, 0.02, args=args)
                i_time.append(t_new)
                i_rate.append(Res['x'].item())
        today_s = datetime.datetime.strftime(today, '%b %d, %Y')
        df_spot.append({'date': today_s, 'rate': i_rate, 'term': term})
    return df_spot

df_spot = spot(dates, bonds, df_bonds, df_prices)


fig, ax = plt.subplots()
for curve in df_spot:
    ax.plot(curve['term'], curve['rate'], label=curve['date'])
plt.legend()
ax.set_title('spot curve')
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels(range(6))
ax.set_xlabel('term')
ax.set_ylabel('spot rate')
plt.show()

def foward(df_spot):
    df_forward=[]
    for curve in df_spot:
        i_time=[]
        i_rate=[]
        length=len(curve['rate'])
        term=curve['term']
        rate=curve['rate']
        cs=CubicSpline(term, rate)
        r1=cs(365)
        for i in range(1, length):
            term=curve['term'][i]
            rate=curve['rate'][i]
            t=term/365
            if t<1: continue
            forward_rate=(rate*t-r1)/(t-1)
            i_time.append(term-365)
            i_rate.append(forward_rate)
        df_forward.append({'date': curve['date'], 'rate': i_rate, 'term': i_time})
    return df_forward

df_forward=foward(df_spot)

fig, ax=plt.subplots()
for curve in df_forward:
    ax.plot(curve['term'], curve['rate'], label=curve['date'])
plt.legend()
ax.set_title('1-year forward curve ')
ax.set_xticks([i * 365 for i in range(5)])
ax.set_xticklabels(range(5))
ax.set_xlabel('term')
ax.set_ylabel('forward rate')
plt.show()

def i_p(df, targets):   
    result=[]
    for curve in df:
        date=curve['date']
        term=curve['term']
        rate=curve['rate']
        cs=CubicSpline(term, rate)
        rate_1=[]
        for target in targets:
            rate_1.append(cs(target).item())
        result.append({date:rate_1})
    return result

def cov(df, targets):   
    targets = 365*np.array(targets)
    df_inter= i_p(df, targets)
    x = []
    for i in range(len(df_inter) - 1):
        r2 = np.array(list(df_inter[i].values())[0])
        r1 = np.array(list(df_inter[i + 1].values())[0])
        x.insert(0, np.log(np.array(r2/r1)))
    X = np.array(x)
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]- np.mean(X[:,i])
    cov = X.transpose()@X
    return cov

targets = [1,2,3,4,5]
cov_ytm = cov(df_ytm, targets)
targets = [1,2,3,4]
cov_forward = cov(df_forward, targets)
print (cov_ytm)
print (cov_forward)

print(np.linalg.eig(cov_ytm))
print(np.linalg.eig(cov_forward))