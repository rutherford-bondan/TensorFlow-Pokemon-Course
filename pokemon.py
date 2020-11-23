import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def dummy_creation(df, dummy_categories):
	for i in dummy_categories:
		df_dummy = pd.get_dummies(df[i])
		df = pd.concat([df, df_dummy], axis=1)
		df = df.drop(i, axis=1)
	return(df)
	
	
def train_test_splitter(DataFrame, column):
	df_train = DataFrame.loc[DataFrame[column] != 1]
	df_test = DataFrame.loc[DataFrame[column] == 1]
	df_train = df_train.drop(column, axis=1)
	df_test = df_test.drop(column, axis=1)
	return(df_train, df_test)
	
def label_delineator(df_train, df_test, label):
	train_data = df_train.drop(label, axis=1).values
	train_labels = df_train[label].values
	test_data = df_test.drop(label,axis=1).values
	test_labels = df_test[label].values
	return(train_data, train_labels, test_data, test_labels)
	
def data_normalizer(train_data, test_data):
	train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
	test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
	return(train_data, test_data)
	
def predictor(model, test_data, test_labels, index):
	prediction = model.predict(test_data)
	if np.argmax(prediction[index]) == test_labels[index]:
		print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
	else:
		print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
		return(prediction)

def main():
	df = pd.read_csv('pokemon_alopez247.csv')
	df.columns
	df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]
	df['isLegendary'] = df['isLegendary'].astype(int)
	df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])
	df_train, df_test = train_test_splitter(df, 'Generation')
	train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')
	train_data, test_data = data_normalizer(train_data, test_data)

	length = train_data.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
	model.add(keras.layers.Dense(2, activation='softmax'))

	model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(train_data, train_labels, epochs=400)

	loss_value, accuracy_value = model.evaluate(test_data, test_labels)
	print(f'Our test accuracy was {accuracy_value}')
	predictor(model, test_data, test_labels, 149) #Mewtwo is number 150
	  
if __name__== "__main__":
	main()
