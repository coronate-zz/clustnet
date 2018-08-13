
import pandas as pd 
import numpy as np 
import pprint as pp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.base          import BaseEstimator , TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline      import FeatureUnion
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

data["categorical_column"] = data["target"].apply( lambda x: "tipo2" if x ==2 else "tipo1")

class abraxasOptimizer( ):
    def __init__(self, data):
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]
        self.substitutionList =  [" ", ".", "/", "(", ")", "-", "%", "'", "[", "]", "{", "}", "=", "?", "__"] 
        self.data =  self.columnRename(data) 

        self.variablesAtipicas = {"numericas":dict(), "categoricas": dict()} #Variables que requieren de revision 
        self.numericDictionary, self.stringDictionary = self.stringsAndNumeric( )

    def stringsAndNumeric( self ):
        """
        Returns two lists:
            numericDictionary: name of numeric Variables with description values
            stringDictionary: name of string Variables with description values
        """
        data = self.data
        numericDictionary = dict()
        stringDictionary = dict()
        for col  in data.columns:
            try:
                numericOperation = np.abs(data[col].get_values()[0])
                numericDictionary[col] = data[col].describe().to_dict() #25% 50% COUNT, MAX, MIN MEAN ...
            except Exception as e:
                print( e )
                stringDictionary[col] = { "categorias" : dict() }
                stringDictionary[col]["total_count"] = data[col].describe().to_dict()["count"]
                stringDictionary[col]["num_unique"] = data[col].describe()["unique"]
                stringDictionary[col]["top_category"] = data[col].describe()["top"]

                for category in data[col].unique():
                    catLen = sum(data[col]==category)
                    stringDictionary[col]["categorias"][category] = catLen

        return numericDictionary, stringDictionary

    def categoryAtipicalEval(self): 
        """Consideramos que todos los strings son variables categoricas.
        para cada categoria nos gustaría saber cuamtas observaciones hay
        y si estas deberían eliminarse, sustituyendose por el categorical
        value más reporatdo"""
        for col in self.stringDictionary.keys():
            self.variablesAtipicas["categoricas"][col] = dict()
            for category in self.data[col].unique():
                catLen = sum(data[col]==category)
                catPct = catLen/self.nrows

                if catLen/self.nrows < .05: #TO DO:  EXISTEN OTRAS PRUEBAS?
                    if catLen/self.nrows < .01: #TO DO:  EXISTEN OTRAS PRUEBAS?
                        print( "\n\t\t  ***WARNING:  EL número de apariciones de la CATEGORIA: {} => {} es menor al 1% del DF ***".format( col, category ))
                    print( "\n\t\t  ***WARNING:  EL número de apariciones de la CATEGORIA: {} => {} es menor al 5% del DF ***".format( col, category ))
                self.variablesAtipicas["categoricas"][col][category] =dict()
                self.variablesAtipicas["categoricas"][col][category]["length_pct"] = catPct
                self.variablesAtipicas["categoricas"][col][category]["length"]     = catLen


    def numericAtipicalEval(self): #TO DO: Aplicar otros criterios de marcado de datos atipicos MEDIANA=> Xi> MEDIANA + ;MEDIANA(Xi-MEDIANA) 
        """Para encontrar las variables atipicas en un conjunto de datos numéricos
        aplicamos una revisión para encontrar el número de observaciones que se 
        encuentran a mas de 1, 2, 2.5 y 3 desviaciones estandar de la media"""

        for col in self.numericDictionary.keys():
            std  = self.numericDictionary[col]["std"]
            mean = self.numericDictionary[col]["mean"]
            max_2std   = (2*std)+ mean
            max_2_5std = (2.5*std)+ mean
            max_3std   = (3*std)+ mean
            max_1std   = std+ mean
            self.variablesAtipicas["numericas"][col] = { "atipical_1std": dict(),  "atipical_2std": dict(), "atipical_2.5std": dict(), "atipical_3std":  dict()} 
            self.variablesAtipicas["numericas"][col]["atipical_1std"]["value"]   = max_1std
            self.variablesAtipicas["numericas"][col]["atipical_2std"]["value"]   = max_2std
            self.variablesAtipicas["numericas"][col]["atipical_2.5std"]["value"] = max_2_5std
            self.variablesAtipicas["numericas"][col]["atipical_3std"]["value"]   = max_3std

            count_1std   = sum(self.data[col] >= self.variablesAtipicas["numericas"][col]["atipical_1std"]["value"] )
            count_2std   = sum(self.data[col] >= self.variablesAtipicas["numericas"][col]["atipical_2std"]["value"] )
            count_2_5std = sum(self.data[col] >= self.variablesAtipicas["numericas"][col]["atipical_2.5std"]["value"])
            count_3std   = sum(self.data[col] >= self.variablesAtipicas["numericas"][col]["atipical_3std"]["value"] )

            self.variablesAtipicas["numericas"][col]["atipical_1std"]["count"]   = count_1std
            self.variablesAtipicas["numericas"][col]["atipical_2std"]["count"]   = count_2std
            self.variablesAtipicas["numericas"][col]["atipical_2.5std"]["count"] = count_2_5std
            self.variablesAtipicas["numericas"][col]["atipical_3std"]["count"]   = count_3std

            
    def categoricalStandarization( self, col, pct ):
        """ Normaliza las variales categóricas convirtiendo las categorias cuya aparicion
        es menor al pct% del total de los datos. Estas variables serán substituidas por 
        la variable categórica con mayor númeor de apariciones

        if len( data[categorical_column] == "cat1"  ) < x% del total de datos 
            => data[categorical_column] == "cat1" is repalced with top categorical value
        """
        for category in data[col].unique():
            lenCategory = len( data[ data[col]== category ] )
            top_category = self.stringDictionary[col][top_category] 

            if (lenCategory/self.nrows) < pct :
                self.data[col].loc[ self.data[col], col   ] = top_category
                #Tenemos que mantener el diccionario actualizado
                del self.stringDictionary[col]["categorias"][category]
                self.stringDictionary[col]["categorias"][top_category][count] += lenCategory   

    def categoricalRename( self ):
    	for col in stringDictionary.keys():
    		for sign in self.substitutionList:
    			self.data[col].str.replace( sign, "_") 


    def columnRename(self, data):
        """Change the format of the columns. All the spaces, points and non alphanumeric values
        are replaced by '_'  """
        colNamesInitial = data.columns
        colNamesTransformed = data.columns 
        colDictionary = dict()

        for i in self.substitutionList: 
        	colNamesTransformed = colNamesTransformed.str.replace( i , "_")
        for i in range(len( colNamesInitial)): 
        	colDictionary[colNamesInitial[i]] =  colNamesTransformed[i]
        data  =  data.rename( columns = colDictionary )

        return data




newObject =  abraxasOptimizer(data) #<= Ejecuta self.stringsAndNumeric( ) que regresa dos diccionarios
								    #con infromacion de las variables categoricas e infromacion de las variables numericas
pp.pprint( newObject.stringDictionary )
pp.pprint( newObject.numericDictionary )

newObject.categoricalRename()       #<= cambia el formato de la variables categoricas
newObject.numericAtipicalEval()     #<= Regresa newObject.variablesAtipicas["numericas"]
pp.pprint( newObject.variablesAtipicas["numericas"])

newObject.categoryAtipicalEval()    #<= Regresa newObject.variablesAtipicas["categoricas"]
pp.pprint( newObject.variablesAtipicas["categoricas"])

#TO DO: Implementar printReporte, newObject.toString....
#TO DO:  Con los diccionaros y los atipical values elegir un criterio para eliminar variables.
#      =>Usar categoricalStandarization  para categoricas.
#	   =>Crear una para numericas.
