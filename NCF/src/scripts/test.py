import numpy as np
import pandas as pd


def test():
	test = [[1, 1], [2, 2], [3, 3]]
	test = pd.DataFrame(test)
	test.to_csv('testLALA.tsv', index=False, sep='\t', header=False)

def generate_test_nfc_data():
	df = pd.read_csv('goodreads_interactions.csv')
	train_data = []
	sample_count = 0
	for user_id in df['user_id'].unique():
		user_data = df[df['user_id'] == user_id]
		pos_items = user_data['book_id'][user_data['rating'] >= 3]
		neg_items = user_data['book_id'][user_data['rating'] < 3]
		if len(neg_items) < 10 or len(pos_items) < 10:
			continue
		train_data.append(user_data[['user_id', 'book_id', 'rating']])
		sample_count += user_data[user_data.columns[0]].count()
		if sample_count >= 60000:
			break
	train_data = pd.concat(train_data)
	train_data.to_csv('book_train.tsv', index=False, sep='\t', header=False)
generate_test_nfc_data()

