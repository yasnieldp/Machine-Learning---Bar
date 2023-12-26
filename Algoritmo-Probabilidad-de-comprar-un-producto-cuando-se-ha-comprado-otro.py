import pandas as pd
data = pd.read_csv("pedidos por mesa.csv", encoding='latin1')

y="Agua 0,5 L"
contingency_table = data.groupby(['command_id', 'product_name']).size().unstack(fill_value=0).reset_index()
data1 = contingency_table.applymap(lambda n: 1 if n > 0 else 0)
a = data1[y].sum()
data2 = data1[(data1[y] > 0)].sum().reset_index()
data3 = data2[(data2[0] > 0)].rename(columns={0: "Cantidad", "product_name": "Nombre_del_Producto"})
data4 = data3[data3["Cantidad"] > 0][1:].sort_values(by='Cantidad', ascending=False).reset_index(drop=True)[1:11]
data4["Probabilidad"] = data4["Cantidad"] / a * 100
data5 = round(data4[data4["Nombre_del_Producto"] != y][["Nombre_del_Producto", "Cantidad", "Probabilidad"]], 2)
print(f"Cantidad de veces que se pidió el producto {y}:", a)
data5.head(14)