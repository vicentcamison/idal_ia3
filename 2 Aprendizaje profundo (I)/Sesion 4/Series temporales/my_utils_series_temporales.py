import numpy as np
import pandas as pd
from IPython.display import display_html


NAN = np.NAN

def enventanar_W_out_1(series, se_saben_antes, W_in=1,
                       serie_target=0):
    n = len(series[0])
    dataX = NAN*np.ones((n,W_in,len(series)))
    if np.sometrue([s.dtype == object for s in series]):
        dataX = dataX.astype(object)
    dataY = series[serie_target].copy()
    
    for i in range(n):
        for j,s in enumerate(se_saben_antes):
            int_s = int(s) # para pasar True a 1 y False a 0
            ini_X = max([0,W_in-i-int_s])
            dataX[i, ini_X:,j] = \
            series[j][max([0,i-W_in+int_s]):min([n,i+int_s])]
    
    return dataX, dataY

def enventanar(series, target, se_saben_antes,
               W_in=1, W_out=1):
    n = len(series[0])
    dataX = NAN*np.ones((n,W_in,len(series)))
    if np.sometrue([s.dtype == object for s in series]):
        dataX = dataX.astype(object)
    if W_out==1:
        dataY = series[target].copy()
    else:
        dataY = NAN*np.ones((n,W_out))
        if series[target].dtype == object:
            dataY = dataY.astype(object)
        dataY[:,0] = series[target].copy()
        for i in range(1,W_out):
            dataY[:-i,i] = dataY[i:,0].copy()
    
    for i in range(n):
        for j,s in enumerate(se_saben_antes):
            int_s = int(s) # para pasar True a 1 y False a 0
            ini_X = max([0,W_in-i-int_s])
            dataX[i, ini_X:,j] = \
            series[j][max([0,i-W_in+int_s]):min([n,i+int_s])]
    
    return dataX, dataY

def my_dfs_display(dfs,names):
    df_styler = []
    for df,n in zip(dfs,names):
        df_styler.append(df.style.set_table_attributes("style='display:inline'").\
                         set_caption(n))
    display_html(df_styler[0]._repr_html_()+"__"+df_styler[1]._repr_html_(),
                 raw=True)

def info_enventanado(X,Y,nombres_series,nombre_target,tiempos=None):
    c0  = '\033[1m'  # empieza negrita
    c1  = '\033[0m'  # termina negrita
    W_in = X.shape[1]
    if len(Y.shape)==1:
        W_out = 1
    else:
        W_out = Y.shape[1]
    print(len(X), "ventanas creadas\n")
    print("X.shape={}".format(X.shape)," Y.shape={}".format(Y.shape),"\n")
    for t in range(len(X)):
        print(c0,"Ventana %d:"%t, c1)
        if tiempos is None:
            nombres_ts = ["t="+str(t+i-W_in) for i in range(W_in)]
            nombres_ts_pred = ["t="+str(t+i) for i in range(W_out)]
        else:
            tiempos = list(tiempos)
            if (t-W_in)<0:
                nombres_ts = ["?"+str(i) for i in range(W_in-t)] + tiempos[:t]
            else:
                nombres_ts = tiempos[(t-W_in):t]
            if (t+W_out-1)>=len(tiempos):
                nombres_ts_pred = tiempos[t:] + ["?"+str(i) for i in range(W_out-(len(tiempos)-t))]
            else:
                nombres_ts_pred = tiempos[t:(t+W_out)]
        aux1 = pd.DataFrame(X[t].T,columns=nombres_ts,index=nombres_series)
        aux2 = pd.DataFrame([Y[t]],columns=nombres_ts_pred,
                            index=[nombre_target])
        if W_out==1:
            my_dfs_display((aux1,aux2),
                           ("X[{}].shape={}".format(t,X[t].shape),
                            "Y[{}]={}".format(t,Y[t])))
        else:
            my_dfs_display((aux1,aux2),
                           ("X[{}].shape={}".format(t,X[t].shape),
                            "Y[{}].shape={}".format(t,Y[t].shape)))


def int2dummy(x, minimo, maximo):
    """Esta función toma un array o lista de enteros y
       construye un numpy array de la misma longitud pero una
       dimensión adicional, que corresponde a las columnas
       con los dummies creados.
       
       minimo y maximo son los valores correspondientes a las
       dos columnas de los extremos.
       
       Ejemplo:
       
            int2dummy([3,0,0,2],2,3) devuelve

            array([[0., 1.],
                   [1., 0.],
                   [1., 0.],
                   [1., 0.]])
                   
    
    """
    salida = np.zeros((len(x), maximo-minimo+1))
    for i in range(len(x)):
        salida[i,x[i]-minimo] = 1
    return salida
