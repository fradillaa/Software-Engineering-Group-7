import pickle
from sklearn.ensemble import RandomForestClassifier # machine learning model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
	
# load pickle file (data adn label) that we saved in create_dataset.py file    
data_dict = pickle.load(open('./data.pickle', 'rb'))

print(f"Jumlah sample: {len(data_dict['data'])}")
print(f"Label unik: {set(data_dict['labels'])}")
# finding the longest data for the model (for normalization)
max_len = max(len(item) for item in data_dict['data']) 

normalized_data = []

# normalizing data size
for item in data_dict['data']:
    if len(item) < max_len:
        item.extend([0] * (max_len - len(item)))
    normalized_data.append(item)

# convert to numpy array for scikit learn processing
data = np.asarray(normalized_data)
labels = np.asarray(data_dict['labels'])

# splitting train and test dataset 80:20
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# training Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# predict and evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# save trained model to file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()