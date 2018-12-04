import pandas as pd
import os as os
import re
import pickle
import random
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from keras.models import load_model
from operator import itemgetter

test_path = "test_federalist\\"
author_articles = list()
temp_articles = list()
test_set1 = list()
outer_set = list()

counter = 0

for author_name in os.listdir(test_path):
	sub_path = test_path + author_name + "\\"
	temp_articles.clear()
	for y in os.listdir(sub_path):
		file_data = open(sub_path + y).read()
		file_data = re.sub('[^a-zA-Z .]+', ' ', file_data)
		file_data = re.sub('[  ]+', ' ', file_data)
		temp_articles.append(file_data)
	author_articles.append([author_name,temp_articles[:]])

for disputed_article in author_articles[0][1]:
	counter = counter + 1
	author_name1 = "disputed_author_" + str(counter)
	article_1 = disputed_article
	for j in range(1,5):
		author_name2 = author_articles[j][0]
		for article_2 in author_articles[j][1]:
			outer_set.append([author_name1,author_name2,article_1,article_2])


# n = 2

# for i in range(0,n*50):
# 	my_range = list(range(0,i)) + list(range(i+1,n*50))
# 	test_set1.clear()
# 	for j in my_range:
# 		author_name1 = author_articles[i][0]
# 		author_name2 = author_articles[j][0]
# 		article1 = author_articles[i][1]
# 		article2 = author_articles[j][1]
# 		test_set1.append([author_name1,author_name2,article1,article2])
# 	outer_set.append(test_set1[:])
	

# # print(outer_set)
# # with open("test_set.txt", "wb") as fp:
# # 	pickle.dump(test_set,fp)


# ########################################################################

with open("tokenizer.sav", "rb") as fp:
	tokenizer = pickle.load(fp)


model = load_model("checkpoints\\1543807102\\lstm_50_50_0.17_0.20.h5")

sim_list = list()



lower_bound = 0
upper_bound = 72

for z in range(0,12):

	test_set = outer_set[lower_bound:upper_bound]

	author1 = list()
	author2 = list()
	sentences1 = list()
	sentences2 = list()

	unzip_test_set = list(zip(*test_set))


	author1 = unzip_test_set[0]
	author2 = unzip_test_set[1]
	sentences1 = unzip_test_set[2]
	sentences2 = unzip_test_set[3]


	test_sentence_pairs = list(zip(sentences1,sentences2))

	# 	is_similar = list()
	# 	is_similar.append(0)
	# 	prev_authors = list()

	# 	final_list = list()

	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

	preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
	results = [(x, y, z) for x, y, z in zip(author1,author2, preds)]
	results.sort(key=itemgetter(2), reverse=True)

	hamilton_sum = 0
	madison_sum = 0
	both_sum = 0
	jay_sum = 0

	hamilton_cnt = 0
	madison_cnt = 0
	both_cnt = 0
	jay_cnt = 0

	for i in range(0,72):
		if results[i][1] == "Hamilton":
			hamilton_cnt = hamilton_cnt + 1
			hamilton_sum = hamilton_sum + results[i][2]
		elif results[i][1] == "Madison":
			madison_cnt = madison_cnt + 1
			madison_sum = madison_sum + results[i][2]
		elif results[i][1] == "Jay":
			jay_cnt = jay_cnt + 1
			jay_sum = jay_sum + results[i][2]
		elif results[i][1] == "Hamilton & Madison":
			both_cnt = both_cnt + 1
			both_sum = both_sum + results[i][2]
	max_avg = list()
	max_avg.append(["Hamilton",hamilton_sum/hamilton_cnt])
	max_avg.append(["Madison",madison_sum/madison_cnt])
	max_avg.append(["Both",both_sum/both_cnt])
	max_avg.append(["Jay",jay_sum/jay_cnt])

	max_similarity = 0.0

	for ii in range(0,4):
		if max_avg[ii][1] > max_similarity:
			max_similarity = max_avg[ii][1]
			result_author = max_avg[ii][0]
	
	print("Hamilton - ", max_avg[0][1])
	print("Madison - ", max_avg[1][1])
	print("Both - ", max_avg[2][1])
	print("Jay - ", max_avg[3][1])

	print("The author is - ", result_author)

	

	lower_bound = lower_bound + 72
	upper_bound = upper_bound + 72

# 	for i in range(0,(n*50)-1):
# 		i_author1 = results[i][0]
# 		i_author2 = results[i][1]
# 		i_similarity = results[i][2]
# 		my_range = list(range(0,i)) + list(range(i+1,(n*50)-1))
# 		temp_similarity = i_similarity
# 		counter = 0
# 		if i_author2 not in prev_authors:
# 			for j in my_range:
# 				j_author1 = results[j][0]
# 				j_author2 = results[j][1]
# 				j_similarity = results[j][2]
# 				if(i_author2 == j_author2):
# 					temp_similarity = temp_similarity + j_similarity
# 					counter = counter + 1

# 			temp_similarity = temp_similarity/counter
# 			final_list.append([i_author1,i_author2,temp_similarity])
# 			prev_authors.append(i_author2)

# 	max_similarity = 0

# 	for i in range(0,n):
# 		if final_list[i][2] > max_similarity:
# 			max_similarity = final_list[i][2]
# 			max_index = i

# 	if(final_list[max_index][0]==final_list[max_index][1]):
# 		correct_author = 1
# 	else:
# 		correct_author = 0

# 	sim_list.append([final_list[max_index],correct_author])

# total = 0
# for i in range(0,n*50):
# 	total = total + sim_list[i][1]
# 	print(sim_list[i])

# print("accuracy = ",(total/(n*50)))