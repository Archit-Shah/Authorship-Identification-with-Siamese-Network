import pandas as pd
import os as os
import re
import pickle
import random
from operator import itemgetter

data = list()
temp = list()
counter = 0
input_path = "data\\"

sentences1 = list()
sentences2 = list()
is_similar = list()
author = list()
counter = 0
input_path = "data\\"
test_path = "test\\"

articles = list()
author_articles = list()


for x in os.listdir(input_path):
	sub_path = input_path + x + "\\"
	author_name = x
	temp.clear()
	for y in os.listdir(sub_path):
		file_data = open(sub_path + y).read()
		file_data = re.sub('[^a-zA-Z .]+', ' ', file_data)
		file_data = re.sub('[  ]+', ' ', file_data)
		# author.append(author_name)
		# articles.append(file_data)
		author_articles.append([author_name,file_data])
		counter = counter + 1
		if counter%2==1:
			sentences1.append(file_data)
		else:
			sentences2.append(file_data)
			is_similar.append(1)
			author.append(author_name)

my_df = list(zip(author,sentences1,sentences2,is_similar))

# print(author_articles[50])

lower_bound = 0
upper_bound = 2499

lower = 0
upper = 49

diff_sentences1 = list()
diff_sentences2 = list()
diff_author = list()
diff_is_similar = list()

train_list = list()

for i in range(0,50):
	for j in range(0,50):
		a_range = range(lower,upper)
		a_index,b_index = random.sample(a_range,2)
		train_list.append(["similar",author_articles[a_index][1],author_articles[b_index][1],1])

	for k in range(0,25):
		a_range = range(lower,upper)
		b_range = list(range(lower_bound,lower)) + list(range(upper,upper_bound))
		a_index = random.choice(a_range)
		b_index = random.choice(b_range)
		train_list.append(["different",author_articles[a_index][1],author_articles[b_index][1],0])

	lower = lower + 50
	upper = upper + 50
		

random.shuffle(train_list)

# counter = 0
# train_set = list()
# test_set = list()

# for x in my_df:
# 	if counter%25 <= 14:
# 		train_set.append(x)
# 	else:
# 		test_set.append(x)
# 	counter = counter + 1

# for x in range(0,len(train_set)-15):
# 	train_set.append(["different",train_set[x][1],train_set[x+15][1],0])

# temp_ind = 0

# for x in range(len(train_set)-15, len(train_set)):
# 	train_set.append(["different",train_set[x][1],train_set[temp_ind][1],0])
# 	temp_ind = temp_ind + 1


# for x in range(0,len(test_set)-10):
# 	test_set.append(["different",test_set[x][1],test_set[x+10][1],0])

# temp_ind = 0

# for x in range(len(test_set)-10, len(test_set)):
# 	test_set.append(["different",test_set[x][1],test_set[temp_ind][1],0])
# 	temp_ind = temp_ind + 1

train_list = list(zip(*train_list))
# test_set = list(zip(*test_set))

# train_sentences1 = train_set[1]
# train_sentences2 = train_set[2]
# train_is_similar = train_set[3]

with open("train_set.txt", "wb") as fp:
	pickle.dump(train_list,fp)

# with open("test_set.txt", "wb") as fp:
# 	pickle.dump(test_set,fp)

# print(len(test_set[1]))

