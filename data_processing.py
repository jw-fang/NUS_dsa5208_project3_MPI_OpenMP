# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:52:35 2026

@author: USER
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
path=r"\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3"
list_=glob(rf"{path}\rainfall\HistoricalRainfallacrossSingapore*.csv")
list_.sort()
concat=[]
for x in tqdm(list_):
    year=Path(x).stem.split('Singapore')[1]
    df=pd.read_csv(x,usecols=['date','timestamp','location_longitude','location_latitude','reading_value'])
    
    df[['_','month','day']]=df['date'].str.split('-',expand=True)
    df.drop(columns=['_','date'],inplace=True)
    df['hour']=df['timestamp'].str.split('T',expand=True)[1].str.split(':',expand=True)[0]
    df[['hour','month','day']]=df[['hour','month','day']].astype(int)
    df.drop(columns=['timestamp'],inplace=True)
    groupby=df.groupby(['location_longitude','location_latitude','hour','month','day']).sum('reading_value')
    groupby['rained']=np.where(groupby['reading_value']>0, 1,0)
    groupby.drop(columns=['reading_value'],inplace=True)
    groupby['year']=year
    concat.append(groupby)
    
groupby_concat=pd.concat(concat)
groupby_concat.reset_index(inplace=True)
groupby_concat['hour']=groupby_concat['hour']/24
groupby_concat['month']=groupby_concat['month']/12
groupby_concat['day']=groupby_concat['day']/31

temp_df=groupby_concat[(groupby_concat['year']=='2017')|(groupby_concat['year']=='2018')]
temp_df.drop(columns=['year'],inplace=True)
temp_df.to_csv(rf'{path}\processed_1.csv',index=False)

print(len(temp_df))
groupby_concat.drop(columns=['year'],inplace=True)
groupby_concat.to_csv(rf'{path}\processed_2.csv',index=False)

print(len(groupby_concat))

len_count=pd.read_csv(rf'{path}\processed.csv',usecols=['rained'])
print(len(len_count))
# %%
path=r"\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3"
df=pd.read_csv(fr'{path}/processed.csv')
df_2=pd.concat([df,df])
df_2.to_csv(fr'{path}/processed_4.csv',index=False)
df_2=pd.concat([df_2,df_2])
df_2.to_csv(fr'{path}/processed_5.csv',index=False)

# %%
#rename
from glob import glob
from pathlib import Path
path=r"\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3"
mpi_glob=glob(fr'{path}/processed*_MPI')
gpu_glob=glob(fr'{path}/processed*_GPU')
full_glob=mpi_glob+gpu_glob
full_glob.sort()

for folder in full_glob:
    size=Path(folder).stem.split('_')[1]
    time_glob=glob(fr'{folder}/*time*.txt')
    for txt in time_glob:
        name=Path(txt).stem
        Path(txt).rename(f'{folder}/{size}_{name}.txt')


# %%

from glob import glob
import pandas as pd
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

pio.renderers.default = 'svg'
path=r'\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3'
folder_glob=glob(fr'{path}\processed*_GPU')
# folder_glob=[r'\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3\processed_*_GPU']
df_concat=[]
df_concat_cold=[]

for time_path in folder_glob:
    time_glob=glob(rf'{time_path}\*time_*.txt')
    time_glob.sort()
    
    for txt in time_glob:
        split=Path(txt).stem.split('_')
        if (len(split)-1)==2: condition=Path(txt).stem.split('_',maxsplit=1)[1]
        else:condition=Path(txt).stem.split('time_',maxsplit=1)[1]
        size=Path(txt).stem[0]
        mpi=condition.split('openmp')[0].split('mpi')[1]
        openmp=condition.split('openmp_')[1]
        
        #cold
        df_cold=pd.read_csv(txt,header=None,nrows=1)
        #warm
        df=pd.read_csv(txt,header=None,skiprows=1)
        for dataframe in [df,df_cold]:
            dataframe.columns=['time']
            dataframe['time']=dataframe['time']
            dataframe['log_time']=np.log(dataframe['time'])
            dataframe['mpi']=int(mpi)
            dataframe['log_mpi']=np.log(dataframe['mpi'])
            dataframe['openmp']=int(openmp)
            dataframe['log_openmp']=np.log(dataframe['openmp'])
            
            if size=='1': dataframe['data_size']=849447
            elif size=='2': dataframe['data_size']=2389537
            elif size=='3': dataframe['data_size']=3548980
            elif size=='4': dataframe['data_size']=7097960
            elif size=='5': dataframe['data_size']=14195920
            dataframe['data_size']=np.log(dataframe['data_size']).round(2)
        
        df['condition']='warm_start'
        df_cold['condition']='cold_start'
        
        for dataframe in [df,df_cold]:
            dataframe['condition_2']='data_size_'+dataframe['data_size'].astype(str)+'_'+dataframe['condition']
            dataframe['condition_3']=np.where(dataframe['openmp']==1,
                                              dataframe['condition']+"_openmp1",
                                              dataframe['condition'])
            
            
            
        df_concat.append(df)
        df_concat_cold.append(df_cold)
    
df_concat=pd.concat(df_concat)
df_concat_cold=pd.concat(df_concat_cold)

#parallel
data_list=np.unique(df_concat['data_size'])
eff_list=[]
eff_cold=[]

for data in data_list:
    df_temp=df_concat[df_concat['data_size']==data]
    ref=df_temp[(df_temp['mpi']==1) 
                & (df_temp['openmp']==1)]['time'].mean()
    df_temp['speed_up']=np.where(
        df_temp['mpi']!=1,
        (ref/df_temp['time']), 1)
    
    if 'GPU' in time_path: divide=72
    else: divide=df_temp['mpi']
    df_temp['eff']=np.where(
        df_temp['mpi']!=1,
       (df_temp['speed_up']/divide), 1)
    eff_list.append(df_temp)

df_concat_eff=pd.concat(eff_list)

for data in data_list:
    df_temp=df_concat_cold[df_concat_cold['data_size']==data]
    ref=df_temp[(df_temp['mpi']==1) 
                & (df_temp['openmp']==1)]['time'].tolist()[0]
    df_temp['speed_up']=np.where(
        df_temp['mpi']!=1,
        ref/df_temp['time'],1)
    
    if 'GPU' in time_path: divide=72
    else: divide=df_temp['mpi']
    
    df_temp['eff']=np.where(
        df_temp['mpi']!=1,
        (df_temp['speed_up']/divide), 1)
    eff_cold.append(df_temp)
    
df_concat_cold_eff=pd.concat(eff_cold)

df_full=pd.concat([df_concat_eff,df_concat_cold_eff])
df_full.reset_index(inplace=True)
df_full.sort_values('data_size', inplace=True)
df_full['log_speed_up']=np.log(df_full['speed_up'])
df_full['log_eff']=np.log(df_full['eff'])
df_full.sort_values('condition_3',inplace=True)

# %%

# scatter

pio.renderers.default = 'svg'

df_full=df_full[(df_full['mpi']>1)|(df_full['openmp']>1)]
condition='warm_start'
for condition in ['warm_start','cold_start']:
    
    px.scatter(df_full[df_full['condition']==condition],
               x="log_openmp",
               y="log_time",
               color='condition_2',
               title='OpenMP threads against log(time)')
    
    px.scatter(df_full[df_full['condition']==condition],
               x="log_mpi",
               y="log_time",
               color='condition_3',
               # trendline='lowess',
               labels={'time':'log(time)','mpi':'log(mpi)'},
               title=f'log(time) against log(MPI), with {condition}').show()
    
    for size in data_list:
        pio.renderers.default = 'svg'
        df_temp=df_full[(df_full['condition']==condition)&
                        (df_full['data_size']==size)]
        px.scatter(df_temp,
                   x="log_mpi",
                   y="log_time",
                   color='condition_3',
                   trendline='lowess',
                   labels={'time':'log(time)','mpi':'log(mpi)'},
                   title=f'log(time) against log(MPI), with data_size {size}').show()
    
    px.scatter(df_full[(df_full['condition']==condition)],
               x="log_mpi",
               y="speed_up",
               color='condition_2',
               trendline='lowess',
               labels={'time':'log(time)','mpi':'log(mpi)'},
               title=f'parallel speedup against log(MPI), with {condition}').show()
        
    px.scatter(df_full[df_full['condition']==condition],
               x="log_mpi",
               y="eff",
               color='condition_2',
               labels={'time':'log(time)','mpi':'log(mpi)'},
               title=f'parallel efficiency ratio against log(MPI), with {condition}').show()
    
    px.scatter(df_full[df_full['condition']==condition],
               x="log_mpi",
               y="log_eff",
               color='condition_2',
               labels={'time':'log(time)','mpi':'log(mpi)'},
               title=f'parallel efficiency (log) against log(MPI), with {condition}').show()
    
    
    

serial=px.scatter(df_full,
           x="data_size",
           y="time",
           trendline="ols",
           color='condition',
           range_y=[0,None],
           labels={'data_size':'lg(data rows)',
                   'time':'log(time)'},
           title='lg(Size of data in rows) against log(time), serial')

serial_r2=px.get_trendline_results(serial).iloc[0][1].rsquared_adj
# %%

#time
pio.renderers.default = 'svg'
heatmap=df_concat_eff.pivot_table(index="openmp",
                                columns="mpi",
                                values="time",
                                aggfunc="mean")

heatmap.index=heatmap.index.astype('str')
heatmap.columns=heatmap.columns.astype('str')

fig=go.Figure(go.Heatmap(
                   x=heatmap.columns,
                   y=heatmap.index,
                   z=heatmap.values,
                   xgap=2,ygap=2,
                   texttemplate="%{z:.2f}"))


fig.update_layout(
    xaxis_title="MPI",
    yaxis_title="log(data_set)")

fig.update_layout(title='Parallel efficiency against MPI processes ')
fig.show()

# 3d

pio.renderers.default = "browser"
d3_plot = px.scatter_3d(
    df_concat_eff,
    x="log_mpi",
    y="log_openmp",
    z="log_time",
    color='condition_2',
    trendline='ols'
    )

d3_plot.show()



# %%
#linear regression
import numpy as np
import statsmodels.api as sm

df_concat=df_concat[(df_concat['mpi']>1)|(df_concat['openmp']>1)]
df_concat['interaction']=df_concat['log_mpi']*df_concat['data_size']
df_concat['interaction2']=df_concat['interaction']*df_concat['interaction']
df_concat['interaction3']=df_concat['log_mpi']*df_concat['data_size']*df_concat['log_openmp']
df_concat['interaction4']=df_concat['interaction3']*df_concat['interaction3']
df_concat['log_mpi2']=df_concat['log_mpi']*df_concat['log_mpi']
df_concat['log_data_size2']=df_concat['data_size']*df_concat['data_size']
df_concat['log_openmp2']=df_concat['log_openmp']*df_concat['log_openmp']
x=df_concat[['log_mpi','log_openmp','data_size','interaction','interaction2','interaction3','interaction4','log_mpi2','log_openmp2','log_data_size2',]]

# for MPI
X=sm.add_constant(x)
model=sm.OLS(df_concat['log_time'],X).fit()
model_fullr2=model.rsquared_adj
parms=model.params
p=model.pvalues


print(model.summary())
results=dict()

# cohn's f2
for feature in x.columns:
    reduced=x.drop(columns=[feature])
    reduced=sm.add_constant(reduced)
    reduced_r2=sm.OLS(df_concat['log_time'],reduced).fit().rsquared_adj
    results[feature]=((model_fullr2-reduced_r2)/(1-reduced_r2))
results=pd.DataFrame({'feature':results.keys(),'value':results.values()}).set_index('feature')
full=pd.concat([parms,p,results],axis=1)
print(results)
# %%

from glob import glob
from pathlib import Path
from pandas import read_csv
path=r'\\wsl.localhost\Ubuntu\home\user\dsa5208_proj3\output_latest\node4'
glob_list=glob(rf'{path}\weights_*.txt')
ref=[x for x in glob_list if 'mpi1openmp_1' in x][0]
ref_df=read_csv(ref,header=None)
ref_df.dropna(axis=1, how='all',inplace=True)
glob_list.remove(ref)
glob_list.sort()

check=dict()

for condition in glob_list:
    name=Path(condition).stem
    check_df=read_csv(condition,header=None)
    check_df.dropna(axis=1, how='all',inplace=True)
    check[name]=((ref_df==check_df).all()).all()

