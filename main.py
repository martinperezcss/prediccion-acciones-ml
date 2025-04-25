# Importacion de librerias necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense


# Cargar e imprimir informacion esencial
df=pd.read_csv('tuarchivo.csv')
print(df.columns)
print(df.info())
print(df.describe())


# Por si acaso tiene valores vacios le pongo el de la fila anterior
df.ffill(inplace=True)


#_________________ Regresion lineal NVIDIA_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_nvd=df[['Rendimiento_Apple',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_NVIDIA',
'Total_activos_NVIDIA', 'ROE_NVIDIA', 'ROA_NVIDIA',
'Apalancamiento_NVIDIA']]
y_nvd=df["Rendimiento_NVIDIA"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_nvd,y_nvd, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


# Creo el modelo
lr=LinearRegression()
# Lo entreno con los datos de entrenamiento
lr.fit(X_train_scl, y_train)


 # Hago la prediccion con los datos de prueba
y_pred=lr.predict(X_test_scl)


#print(y_pred)


# Calculo el RMSE y r cuarado
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2=r2_score(y_test,y_pred)
print("RMSE: ",rmse)
print("R cuadrado: ",r2)


'''Un gráfico con una línea para indicar los valores reales y los puntos que␣
comparan los valores reales y predichos'''
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test, color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Regresion lineal NVIDIA")
plt.show()

# Calculo de residuos
res=y_test-y_pred
# Grafica de residuos
plt.scatter(y_pred, res)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.show()


# Histograma de residuos
plt.hist(res)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')
plt.show()

#_________________ Regresion lineal Apple_____________________
# Defino las variables independientes en x y la variable a predecir en y
X_ap=df[['Rendimiento_NVIDIA',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Apple', 'Total_activos_Apple',
'ROE_Apple', 'ROA_Apple', 'Apalancamiento_Apple']]
y_ap=df["Rendimiento_Apple"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_ap,y_ap, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


# Creo el modelo
lr=LinearRegression()
# Lo entreno con los datos de entrenamiento
lr.fit(X_train_scl, y_train)

# Hago la prediccion con los datos de prueba
y_pred=lr.predict(X_test_scl)
#print(y_pred)


# Calculo el RMSE y r cuarado
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2=r2_score(y_test,y_pred)
print("RMSE: ",rmse)
print("R cuadrado: ",r2)


'''Un gráfico con una línea para indicar los valores reales y los puntos que␣
comparan los valores reales y predichos'''
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test, color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Regresion lineal NVIDIA")
plt.show()


# Calculo de residuos
res=y_test-y_pred
# Grafica de residuos
plt.scatter(y_pred, res)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.show()


# Histograma de residuos
plt.hist(res)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')
plt.show()

#_________________ Regresion lineal Meta_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_mt=df[[
'Rendimiento_NVIDIA','Rendimiento_Apple', 'Rendimiento_VIX','Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Meta',
'Total_activos_Meta', 'ROE_Meta', 'ROA_Meta', 'Apalancamiento_Meta']]
y_mt=df["Rendimiento_Meta_Platforms"]

# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_mt,y_mt, test_size=0.3)

'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


 # Creo el modelo
lr=LinearRegression()
# Lo entreno con los datos de entrenamiento
lr.fit(X_train_scl, y_train)


# Hago la prediccion con los datos de prueba
y_pred=lr.predict(X_test_scl)
#print(y_pred)



 # Calculo el RMSE y r cuarado
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2=r2_score(y_test,y_pred)
print("RMSE: ",rmse)
print("R cuadrado: ",r2)



'''Un gráfico con una línea para indicar los valores reales y los puntos que␣
comparan los valores reales y predichos'''
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test, color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Regresion lineal NVIDIA")
plt.show()


 # Calculo de residuos
res=y_test-y_pred
# Grafica de residuos
plt.scatter(y_pred, res)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.show()




 # Histograma de residuos
plt.hist(res)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')
plt.show()


#_________________ Regresion logistica_____________________
# Transformacion a variable categórica binaria

# Transformar variables dependientes a binarias
df["Rendimiento_NVIDIA_bin"]=(df["Rendimiento_NVIDIA"]>0).astype(int)
df["Rendimiento_Apple_bin"]=(df["Rendimiento_Apple"]>0).astype(int)
df["Rendimiento_Meta_Platforms_bin"]=(df["Rendimiento_Meta_Platforms"]>0).astype(int)
#print(df.head())
#_________________ Regresion logistica NVIDIA_____________________
# Defino las variables independientes en x y la variable a predecir en y
X_nvd=df[['Rendimiento_Apple',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_NVIDIA',
'Total_activos_NVIDIA', 'ROE_NVIDIA', 'ROA_NVIDIA',
'Apalancamiento_NVIDIA']]
y_nvd=df["Rendimiento_NVIDIA_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_nvd,y_nvd, test_size=0.3)

'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


# Crear el modelo
lor=LogisticRegression()
# Entrenarlo
lor.fit(X_train_scl,y_train)


 # Realizo las predicciones
y_pred=lor.predict(X_test_scl)


# Calculo de prediciones
# Como se calcula con sklearn
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# Como se calculo en clase
accuracy_manual = 1-((y_pred != y_test).mean())
print(accuracy_manual)
# A partir de ahora lo hare con sklearn, ya que me parece mucho mas practico


# Matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


#_________________ Regresion logistica Apple_____________________
# Defino las variables independientes en x y la variable a predecir en y
X_ap=df[['Rendimiento_NVIDIA',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Apple', 'Total_activos_Apple',
'ROE_Apple', 'ROA_Apple', 'Apalancamiento_Apple']]
y_ap=df["Rendimiento_Apple_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_ap,y_ap, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())

# Crear el modelo
lor=LogisticRegression()
# Entrenarlo
lor.fit(X_train_scl,y_train)


 # Realizo las predicciones
y_pred=lor.predict(X_test_scl)

 # Calculo de prediciones
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# Matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#_________________ Regresion logistica Meta_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_mt=df[[
'Rendimiento_NVIDIA','Rendimiento_Apple', 'Rendimiento_VIX','Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Meta',
'Total_activos_Meta', 'ROE_Meta', 'ROA_Meta', 'Apalancamiento_Meta']]
y_mt=df["Rendimiento_Meta_Platforms_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_mt,y_mt, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


 # Crear el modelo
lor=LogisticRegression()
# Entrenarlo
lor.fit(X_train_scl,y_train)


 # Realizo las predicciones
y_pred=lor.predict(X_test_scl)


# Calculo de prediciones
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# Matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#_________________Arboles de decision NMVIDIA_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_nvd=df[['Rendimiento_Apple',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_NVIDIA',
'Total_activos_NVIDIA', 'ROE_NVIDIA', 'ROA_NVIDIA',
'Apalancamiento_NVIDIA']]
y_nvd=df["Rendimiento_NVIDIA"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_nvd,y_nvd, test_size=0.3)


# Creo el modelo
rf = RandomForestRegressor()
# Lo entreno
rf.fit(X_train,y_train)


# Hago la prediccion
y_pred=rf.predict(X_test)

 # Calcular RMSE
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# Calcular r cuadrado
r2=r2_score(y_test,y_pred)
print(r2)


#_________________Arboles de decision Apple_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_ap=df[['Rendimiento_NVIDIA',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Apple', 'Total_activos_Apple',
'ROE_Apple', 'ROA_Apple', 'Apalancamiento_Apple']]
y_ap=df["Rendimiento_Apple"]


# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_ap,y_ap, test_size=0.3)

# Creo el modelo
rf = RandomForestRegressor()
# Lo entreno
rf.fit(X_train,y_train)


# Hago la prediccion
y_pred=rf.predict(X_test)


# Calcular RMSE
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# Calcular r cuadrado
r2=r2_score(y_test,y_pred)
print(r2)


#_________________Arboles de decision Meta_____________________
# Defino las variables independientes en x y la variable a predecir en y
X_mt=df[[
'Rendimiento_NVIDIA','Rendimiento_Apple', 'Rendimiento_VIX','Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Meta',
'Total_activos_Meta', 'ROE_Meta', 'ROA_Meta', 'Apalancamiento_Meta']]
y_mt=df["Rendimiento_Meta_Platforms"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_mt,y_mt, test_size=0.3)


# Creo el modelo
rf = RandomForestRegressor()
# Lo entreno
rf.fit(X_train,y_train)

# Hago la prediccion
y_pred=rf.predict(X_test)


# Calcular RMSE
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# Calcular r cuadrado
r2=r2_score(y_test,y_pred)
print(r2)


#_________________SVM NVIDIA_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_nvd=df[['Rendimiento_Apple',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500','Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_NVIDIA',
'Total_activos_NVIDIA', 'ROE_NVIDIA', 'ROA_NVIDIA',
'Apalancamiento_NVIDIA']]
y_nvd=df["Rendimiento_NVIDIA_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_nvd,y_nvd, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())

# Creo el modelo
svm=SVC()
# Lo entreno
svm.fit(X_train_scl,y_train)


# Hago las predicciones
y_pred=svm.predict(X_test_scl)


# Calculo precision
prec=accuracy_score(y_test,y_pred)
print(prec)
# Matriz de confusion
cm=confusion_matrix(y_test,y_pred)
print(cm)

#_________________SVM Apple_____________________	


# Defino las variables independientes en x y la variable a predecir en y
X_ap=df[['Rendimiento_NVIDIA',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40','Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Apple', 'Total_activos_Apple',
'ROE_Apple', 'ROA_Apple', 'Apalancamiento_Apple']]
y_ap=df["Rendimiento_Apple_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_ap,y_ap, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())

# Creo el modelo
svm=SVC()
# Lo entreno
svm.fit(X_train_scl,y_train)


# Hago las predicciones
y_pred=svm.predict(X_test_scl)


# Calculo precision
prec=accuracy_score(y_test,y_pred)
print(prec)
# Matriz de confusion
cm=confusion_matrix(y_test,y_pred)
print(cm)

#_________________SVM Meta_____________________


# Defino las variables independientes en x y la variable a predecir en y
X_mt=df[['Rendimiento_NVIDIA','Rendimiento_Apple', 'Rendimiento_VIX','Rendimiento_SP_500','Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Meta',
'Total_activos_Meta', 'ROE_Meta', 'ROA_Meta', 'Apalancamiento_Meta']]
y_mt=df["Rendimiento_Meta_Platforms_bin"]
# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_mt,y_mt, test_size=0.3)


'''Normalizo las variables independientes, ya que pueden estar en escalas␣
diferentes, en el train utilizo fit para que los parametros se ajusten a␣
estos datos'''
scl=StandardScaler()
X_train_scl=scl.fit_transform(X_train)
X_test_scl=scl.transform(X_test)
#print(X_train.head())
#print(X_test.head())


# Creo el modelo
svm=SVC()
# Lo entreno
svm.fit(X_train_scl,y_train)


# Hago las predicciones
y_pred=svm.predict(X_test_scl)


# Calculo precision
prec=accuracy_score(y_test,y_pred)
print(prec)
# Matriz de confusion
cm=confusion_matrix(y_test,y_pred)
print(cm)



#_________________LSTM NVIDIA_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_nvd=df[['Rendimiento_Apple',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_NVIDIA',
'Total_activos_NVIDIA', 'ROE_NVIDIA', 'ROA_NVIDIA',
'Apalancamiento_NVIDIA']]
y_nvd=df["Rendimiento_NVIDIA"]


 # Normalizar los datos
scl=MinMaxScaler()
X_scl=scl.fit_transform(X_nvd)


 # Funcion para hacer secuencias de los datos, ya que segun he visto este modelo lo requiere
def secu(X,y,pasos=10):
    # Listas vacias para cada variable
    X_s,y_s=[], []
    # Bucle para las secuencias
    for i in range(len(X)-pasos):
        # Para dividir las filas en secuencias
        X_s.append(X[i:i+pasos])
        # Asignarle un valor a cada secuencia
        y_s.append(y[i+pasos])
    return np.array(X_s), np.array(y_s)


# Crear las secuencias con las variables
X_s,y_s=secu(X_scl,y_nvd)
#print(X_s)
#print(y_s)


# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_s, y_s, test_size=0.3)


m=Sequential([
# Defino la entrada, 10 para numero de secuencias creadas, 1 para longitud de secuencia y 2 para numero de caracteristicas
Input(shape=(10, X_train.shape[2])),
LSTM(10),
Dense(1)
])


# Compilo el modelo
m.compile(optimizer="adam", loss="mse")


# Entreno al modelo
m.fit(X_train, y_train)



# Miro la evaluacion del modelo, ya que MSE se pone en la compilacion del modelo, por tanto el resultado sera el MSE
print(m.evaluate(X_test,y_test))
# Imprimir RMSE
print(np.sqrt(m.evaluate(X_test,y_test)))

# Predecir los resultados

y_pred=m.predict(X_test)
print(y_pred)


# Para ver el error en las predicciones
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


# Grafica de residuos
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test ,color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs predichos")
plt.show()


#_________________LSTM Apple_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_ap=df[['Rendimiento_NVIDIA',
'Rendimiento_Meta_Platforms', 'Rendimiento_VIX', 'Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Apple', 'Total_activos_Apple',
'ROE_Apple', 'ROA_Apple', 'Apalancamiento_Apple']]
y_ap=df["Rendimiento_Apple"]


# Normalizar los datos
scl=MinMaxScaler()
X_scl=scl.fit_transform(X_ap)


# Crear las secuencias con las variables
X_s,y_s=secu(X_scl,y_ap)
#print(X_s)
#print(y_s)


# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_s, y_s, test_size=0.3)


m=Sequential([
# Defino la entrada, 10 para numero de secuencias creadas, 1 para longitud de secuencia y 2 para numero de caracteristicas
Input(shape=(10, X_train.shape[2])),
LSTM(10),
Dense(1)
])

# Compilo el modelo
m.compile(optimizer="adam", loss="mse")


# Entreno al modelo
m.fit(X_train, y_train)


# Miro la evaluacion del modelo, ya que MSE se pone en la compilacion del modelo, por tanto el resultado sera el MSE
print(m.evaluate(X_test,y_test))
# Imprimir RMSE
print(np.sqrt(m.evaluate(X_test,y_test)))

# Predecir los resultados
y_pred=m.predict(X_test)
print(y_pred)

# Para ver el error en las predicciones
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


# Calcular r cuadrado
r2=r2_score(y_test,y_pred)
print(r2)


 # Grafica de residuos
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test ,color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs predichos")
plt.show()


#_________________LSTM Meta_____________________

# Defino las variables independientes en x y la variable a predecir en y
X_mt=df[[
'Rendimiento_NVIDIA','Rendimiento_Apple', 'Rendimiento_VIX','Rendimiento_SP_500',
'Rendimiento_DAX', 'Rendimiento_FTSE_100', 'Rendimiento_CAC_40',
'Rendimiento_IBEX_35', 'Rendimineto_NIKKEI_225',
'Rendimiento_NASDAQ_100', 'Redimiento_EUR_USD', 'Sentimiento',
'Rendimiento_sentimiento', 'Crecimiento_RN_Meta',
'Total_activos_Meta', 'ROE_Meta', 'ROA_Meta', 'Apalancamiento_Meta']]
y_mt=df["Rendimiento_Meta_Platforms"]


# Normalizar los datos
scl=MinMaxScaler()
X_scl=scl.fit_transform(X_mt)


# Crear las secuencias con las variables
X_s,y_s=secu(X_scl,y_mt)
#print(X_s)
#print(y_s)


# Separo datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X_s, y_s, test_size=0.3)

m=Sequential([
# Defino la entrada, 10 para numero de secuencias creadas, 1 para longitud de secuencia y 2 para numero de caracteristicas
Input(shape=(10, X_train.shape[2])),
LSTM(10),
Dense(1)
])

# Compilo el modelo
m.compile(optimizer="adam", loss="mse")

# Entreno al modelo
m.fit(X_train, y_train)


# Miro la evaluacion del modelo, ya que MSE se pone en la compilacion del modelo, por tanto el resultado sera el MSE
print(m.evaluate(X_test,y_test))
# Imprimir RMSE
print(np.sqrt(m.evaluate(X_test,y_test)))

# Predecir los resultados
y_pred=m.predict(X_test)
#print(y_pred)


# Para ver el error en las predicciones
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


# Calcular r cuadrado
r2=r2_score(y_test,y_pred)
print(r2)


# Grafica de residuos
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_test ,color='black')
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs predichos")
plt.show()