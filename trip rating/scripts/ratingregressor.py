import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
with open('RF-Regrassion-Rating.joblib','rb') as file:
    RF=joblib.load(file)
with open('TFIDFDTregression.joblib','rb') as file:
    tfidf=joblib.load(file)
le=LabelEncoder()
activity=str(input('enter the trip activity'))
price=str(input('enter the trip budget'))
location=str(input('enter the trip location'))
df=pd.DataFrame({'Activity':[activity],'Price':[price],'location':[location]},columns=['Activity','Price','location'])
df["location"]=le.fit_transform(df["location"])
df["Price"]=le.fit_transform(df["Price"])
activities=tfidf.transform(df["Activity"])
df2=pd.DataFrame(activities.toarray(),columns=tfidf.get_feature_names_out())
df3=pd.concat([df[["location","Price"]],df2],axis=1)
prediction=RF.predict(df3)
print(f'the overall trip rating is {prediction[0]:.2f}')
    