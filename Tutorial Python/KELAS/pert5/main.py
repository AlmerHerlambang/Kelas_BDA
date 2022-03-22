# import cuaca
import pandas as pd
dataku = pd.read_csv('cuaca.csv')

dataku.info()

x=dataku[dataku.columns[:-1]]
y=dataku[dataku.columns[-1]]

print("\n Fitur:\n", x)
print("\n Target:\n", y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in dataku.columns.values:
    if dataku[col].dtype=='bool':
        data=dataku[col].append(dataku[col])
        le.fit(data.values)
        dataku[col]=le.transform(dataku[col])
dataku.head(10)

dataku.to_csv('cuaca_ubah.csv', header=True, index=False)