import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("cuentas.csv")
data['created_at'] = pd.to_datetime(data['created_at']).dt.date
data1 = data.groupby('created_at')['price'].sum().reset_index()
data1.columns = ['ds', 'y']
data1 = data1.sort_values(by='ds')
data2 = data1.iloc[195:948] #predecir desde 1-1-2021 a 30-12-2022  ----- 2 años
model = Prophet()
model = Prophet(seasonality_mode='multiplicative')
model.add_seasonality(name='year', period=365, fourier_order=1)
model.fit(data2)

start_date = '2023-06-01'
end_date = '2023-10-05'
future = pd.date_range(start=start_date, end=end_date, freq='D')
future = pd.DataFrame({'ds': future})
forecast = model.predict(future)
predictions_df = forecast[['ds', "yhat"]].rename(columns={'ds': 'Fecha', 'yhat': 'Predicción'})
predictions_df['Fecha'] = pd.to_datetime(predictions_df['Fecha']).dt.date

data_main = pd.merge(left=predictions_df, right=data1, how="inner", left_on="Fecha", right_on="ds")
data_main.drop(columns=['ds'], inplace=True)
data_main['diferencia'] = data_main['Predicción'] - data_main['y']
data_main['diferencia'] = data_main['diferencia'].abs()
data_main=data_main[["Fecha","y","Predicción","diferencia"]].rename(columns={'y': 'Importe'})
data_main=round(data_main, 2)

y_true = data_main['Importe']
y_pred = data_main['Predicción']
mae = round(mean_absolute_error(y_true, y_pred), 2)
mape = round(np.mean(np.abs((y_true - y_pred) / y_true) * 100), 2)

print("Predicción de Ventas")
print("Error Absoluto Medio (MAE):", mae)
print("Error Porcentual Absoluto Medio (MAPE):", mape)
data_main.head(10)