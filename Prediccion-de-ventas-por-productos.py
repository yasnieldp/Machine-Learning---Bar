import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import math
data = pd.read_csv("pedidos por mesa.csv", encoding='latin1')
producto = "Agua 0,5 L"
data['created_at'] = pd.to_datetime(data['created_at']).dt.date
data1 = data.iloc[2001:81669]
data2 = data1[data1['product_name'] == producto]
data3 = data2.groupby('created_at')['qty'].sum().reset_index()
data3.columns = ['ds', 'y']
data4 = data3.sort_values(by='ds')

model = Prophet()
model = Prophet(seasonality_mode='multiplicative')
model.add_seasonality(name='year', period=365, fourier_order=1)
model.fit(data4)

start_date = '2023-06-01'
end_date = '2023-10-05'
future = pd.date_range(start=start_date, end=end_date, freq='D')
future = pd.DataFrame({'ds': future})
forecast = model.predict(future)
predictions_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Fecha', 'yhat': 'Predicción'})
predictions_df['Predicción'] = predictions_df['Predicción'].apply(math.floor)

data5 = data[data['product_name'] == producto]
data6 = data5.groupby('created_at')['qty'].sum().reset_index()
data6['created_at'] = pd.to_datetime(data6['created_at'])

data_main = pd.merge(left=predictions_df, right=data6, how="inner", left_on="Fecha", right_on="created_at")
data_main.drop(columns=['created_at'], inplace=True)
data_main['diferencia'] = (data_main['Predicción'] - data_main['qty']).abs()
y_true = data_main['qty']
y_pred = data_main['Predicción']
mae = round(mean_absolute_error(y_true, y_pred),2)

print("Predicción de ventas por producto")
print("Nombre del Producto:", producto)
print("Error Absoluto Medio (MAE):", mae)
data_main.head(10)