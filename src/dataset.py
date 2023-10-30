import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from functools import reduce
import json


exclude_features_names = []

targets = []
features = []
for feature in pd.read_excel('./clanci/realestate_prices/data/covariates.xlsx').to_dict(orient='records'):
    feature['type'] = int if feature['type'] == 'int' else float if feature['type'] == 'float' else str
    if feature['group'] == 'target':
        targets.append(feature)
        continue
    else:
        features.append(feature)


feature_names = [x['name'] for x in features]
target_names = [x['name'] for x in targets]
data = pd.read_excel('./clanci/realestate_prices/data/hp_ljubljana_new_with_rooms.xlsx')[feature_names + target_names]


categorical_features = [x for x in features if  (x['type'] == str) and (x not in exclude_features_names)]
categorical_features_names = [x['name'] for x in categorical_features]
numerical_features = [x for x in features if  (x['type'] in (int, float)) and (x not in exclude_features_names)]
numerical_features_names = [x['name'] for x in numerical_features]

# target = {'name': 'price_m2', 'type': float}

onehot = OneHotEncoder(sparse=False)
onehot.fit(data[categorical_features_names])
encoded_features = list(reduce(lambda x,y: x + y, [[{"name": f"{feature['name']}_{cat}", "type": int, "group": feature['group']} for cat in cats] for cats, feature in zip(onehot.categories_, categorical_features)]))
encoded_features_names = [x['name'] for x in encoded_features]

X_encoded = onehot.transform(data[categorical_features_names])
X_numerical = data[numerical_features_names]
X = pd.DataFrame(columns=numerical_features_names + encoded_features_names)


X[numerical_features_names] = X_numerical
X[encoded_features_names] = X_encoded
X = X.astype({x['name'] : x['type'] for x in numerical_features + encoded_features})
if len(target_names)==1:
    y = data[target_names[0]]
    y = y.astype({targets[0]['name'] : targets[0]['type']})
else:
    y = data[target_names]
    y = y.astype({x['name'] : x['type'] for x in targets})


with open('./clanci/realestate_prices/data/numerical_features.json', 'w') as f:
    json.dump(numerical_features,f, default = str)
    
with open('./clanci/realestate_prices/data/categorical_features.json', 'w') as f:
    json.dump(categorical_features,f, default = str)

with open('./clanci/realestate_prices/data/encoded_categorical_features.json', 'w') as f:
    json.dump(encoded_features,f, default = str)
    
X.to_excel('./clanci/realestate_prices/data/features.xlsx', index=False)
y.to_excel('./clanci/realestate_prices/data/target.xlsx', index=False)
