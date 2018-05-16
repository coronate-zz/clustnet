from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
class preprorec:
    '''
    Preprocess of the data for a simple recommender system
    INPUTS
    df: pandas.DataFrame a dataframe
    column: str Name of the column
    treshold: int or float  a treshold to take a value as an outlier
    tratneg: bool True for no consider the route 900
    '''
    def __init__(self, df, column, treshold):
        '''
        Change the negative values for zeros
        '''
        self.column = column
        self.treshold = treshold
        
        df[column] = np.where(df[column]<0,0,df[column])
        self.data = df
                    
        # Detection of outliers
        ## Creo que esta medida puede ser cambiada en un contexto de desviaciones std. El treshhold puedes ser subjetivo
        column_media = np.median(self.data[self.column])
        column_minusMedia =np.median(np.abs(self.data[self.column]-column_media))
        self.outliers = self.data.index[np.abs(self.data[self.column] - column_media)/column_minusMedia > self.treshold].tolist()

        """
        column_mean = np.mean(self.data[self.column]) #MEAN OF UNITS SOLD
        column_dstd = np.std( self.data[self.column] )
        self.outliers = self.data.index[np.abs( self.data[self.column] - column_media )  > (self.treshold * column_dstd ].tolist()
        """

        #Transform the outliers in 0 or 2
        self.bad = self.data.ix[self.outliers, :] #VALORES OUTLIERS
        self.good = self.data[~self.data.VentaUni.isin(self.bad.VentaUni.unique())]
        self.min = self.good[self.column].min()
        self.bad['like'] = np.where(self.bad[self.column] > self.min, 2, 0)

        """
        #Transform the outliers in 0 or 2
        self.bad = self.data.ix[self.outliers, :] #VALORES OUTLIERS
        self.good = self.data[~self.data.VentaUni.isin(self.bad.VentaUni.unique())]
        self.min = self.good[self.column].min() #MINIMO DE SOLO LAS OBSERVACIONES BUENAS
        self.bad['like'] = np.where(self.bad[self.column] > self.min, 2, 0)
        """

        #Transform the good data in 0 or 1
        scaler = MinMaxScaler()
        scaler.fit(np.asarray(self.good.VentaUni).reshape(-1,1))
        good_normval = scaler.transform(np.asarray(self.good[self.column]).reshape(-1,1))
        good_normval_mean = good_normval.mean()
        like = np.where(good_normval > good_normval_mean, 1, 0)
        self.good['like'] = like #good["like"] ==  1 si es mayor a la media, 0 otherwise
     
    def output(self):
        '''
        The complete product df with like or not
        '''
        self.complete = pd.concat([self.good, self.bad])
        return(self.complete)