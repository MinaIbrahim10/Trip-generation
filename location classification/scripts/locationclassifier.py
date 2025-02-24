import joblib
import pandas as pd
with open('DTclassification.joblib','rb') as file:
    DT=joblib.load(file)
with open('TFIDFDTclassification.joblib','rb') as file:
    tfidf=joblib.load(file)
description=input('enter your description')
df=pd.DataFrame({'description':[description]},columns=['description'])
descriptions=tfidf.transform(df['description'])
df2=pd.DataFrame(descriptions.toarray(),columns=tfidf.get_feature_names_out())
prediction=DT.predict(df2)
print(f'the location based on the description is {prediction[0]}')