import pandas as pd

def seasonal_avg(df, h, id_col="unique_id", time_col="date", target_col="sales"):
    out=[]
    for uid,g in df.groupby(id_col):
        last=g.tail(7)[target_col].mean()
        fut=pd.date_range(g[time_col].max()+pd.Timedelta(days=1),periods=h)
        out.append(pd.DataFrame({id_col:uid,time_col:fut,"yhat":last,"model":"seasonal_avg"}))
    return pd.concat(out)
