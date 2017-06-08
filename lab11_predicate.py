import sys
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from datetime import datetime

eventtype=['accidentsAndIncidents','roadwork','precipitation','deviceStatus','obstruction','trafficConditions']
date_format="%Y-%m-%d"
num_partitions=100
df=pd.DataFrame()

def deltaday(date1, date2):
	time1=datetime.strptime(date1,date_format)
	time2=datetime.strptime(date2,date_format)
	return (time2-time1).days

def predicateEvent(partData):
    global df
    pred_arr=partData[:].values
    lens=len(pred_arr)
    res=[[0 for x in range(6)] for y in range(lens)]
    for index in range(lens):
        row=pred_arr[index]
        event=df[(df['latitude']>=min(row[0],row[2]))&
             (df['latitude']<=max(row[0],row[2]))&
             (df['longitude']>=min(row[1],row[3]))&
             (df['longitude']<=max(row[3],row[1]))]
        group=event.groupby('event_type')
        interval=deltaday(row[4],row[5])
        for i in range(6):
            hashset=set(group.groups.keys())
            if(eventtype[i] not in hashset):
                continue
            count=group.get_group(eventtype[i]).groupby('closed_tstamp').size()
            if(len(count)<2):
                continue
            x=[[x] for x in count[:len(count)-1]]
            y=count[1:]
            lm=LinearRegression()
            lm.fit(x,y)
            gap=int(str(row[4])[:4])-int(group.get_group(eventtype[i])['closed_tstamp'].max())
            prediction=y[len(y)-1]
            for j in range(gap):
                prediction=lm.predict(prediction)[0]
            res[index][i]=max(0,int(prediction*interval/365))
    res_dataframe= pd.DataFrame(res)
    res_dataframe.columns = ['A','R','P','D','O','T']
    #res_dataframe['id']=partData['trial_id']
    return res_dataframe


def main(path):
    global df
    df=pd.read_csv(path+"/events_train.tsv",sep='\t',na_values=['-'])
    pred=pd.read_csv(path+"/t.tsv",sep='\t',na_values=['-'])
    df=df[['closed_tstamp','event_type','latitude','longitude']]
    df['closed_tstamp']=df['closed_tstamp'].str.extract('(....-..-..)', expand=True)
    df['closed_tstamp']=df['closed_tstamp'].map(lambda x: str(x)[:4])
    pred['start']=pred['start'].str.extract('(....-..-..)', expand=True)
    pred['end']=pred['end'].str.extract('(....-..-..)', expand=True)
    del pred['trial_id']
    pred_split = np.array_split(pred, num_partitions)
    pool = mp.Pool(mp.cpu_count()-1)
    result = pd.concat(pool.map(predicateEvent, pred_split))
    pool.close()
    pool.join()
    output_path=os.getcwd()+'/predict.event4.txt'
    result.to_csv(output_path, sep='\t', index=False, header=None)
    
if __name__=="__main__":
	main(sys.argv[1])