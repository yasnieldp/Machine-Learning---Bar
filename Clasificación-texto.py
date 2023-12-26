import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

data = pd.read_excel("train.xlsx")
x=data["texto"]
y=data["clase"]
data1=pd.read_excel("Test.xlsx")
xx=data1["texto"]
yy=data1["clase"]
model = make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(x,y)
ypred=model.predict(xx)
print(classification_report(yy,ypred))

comentario=("Una mala experiencia, pedido a domicilio con más de 45 minutos de espera para una margarita y unas patatas fritas, las cuales han llegado quemadas y parecían estar cocidas en aceite, la pizza un poco mejor pero nada del otro mundo.")

predict=model.predict([comentario])
print("-----------------------------------------------------")
if predict == 1:
    print("Comentario Positivo")
else:
    print("Comentario Negativo")