Movie Recommender Systems
==============================
# md
Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'video',
       'vote_average', 'vote_count'],
      dtype='object')

df=pd.DataFrame(md['genres'].head(5).values) # split by 'genres'

# qualified
Index(['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres',
       'wr'],
      dtype='object')

 	title 	    year 	vote_count 	vote_average 	popularity 	genres
0 	Toy Story 	1995 	5415 	              7 	21.946943 	[Animation, Comedy, Family]
1 	Jumanji 	1995 	2413 	              6 	17.015539 	[Adventure, Fantasy, Family]
5 	Heat 	    1995 	1886 	              7 	17.924927 	[Action, Crime, Drama, Thriller]
9 	GoldenEye 	1995 	1194 	              6 	14.686036 	[Adventure, Action, Thriller]
15 	Casino 	    1995 	1343 	              7 	10.137389 	[Drama, Crime]

# s
0        Animation
0           Comedy
0           Family
1        Adventure
1          Fantasy
1           Family
           ...    
45461       Family
45462        Drama
45463       Action
45463        Drama
45463     Thriller
Name: genre, Length: 91106, dtype: object

#gen_md = old md with out 'genres' + new 'genres' s

Index(['adult', 'belongs_to_collection', 'budget', 'homepage', 'id', 'imdb_id',
       'original_language', 'original_title', 'overview', 'popularity',
       'poster_path', 'production_companies', 'production_countries',
       'release_date', 'revenue', 'runtime', 'spoken_languages', 'status',
       'tagline', 'title', 'video', 'vote_average', 'vote_count', 'year',
       'genre'],
      dtype='object')

Movie Description Based Recommender
=========================================

# smd
Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'video',
       'vote_average', 'vote_count', 'year'],
      dtype='object')

x=tfidf_matrix[0,:].nonzero() # (2, 55)
(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32),
 array([ 14374, 135999,  78094,  70735, 173370, 263028,  32156, 211426,
         40915,  32149, 182227, 263030, 108388,  10561, 180445, 143950,
          6282, 207239, 139595,  32153,  29230,  24049,  10548, 203273,
        106359, 140969, 242500,  10578, 263023, 136575,  63703,  14372,
        135993,  78024,  70725, 173335, 211425,  40897, 182226, 108234,
        180436, 143903,   6268, 207235, 139593,  32147,  29222,  24045,
        203269, 106340, 140882, 242493,  10545, 263021, 136477],
       dtype=int32))

features = tf.get_feature_names() # list of all the tokens or n-grams or words. 
features.sort(key=features.count,reverse=True)
['00',
 '00 agent',
 '00 body',
 '00 middle',
 '000',
 '000 000',
 '000 50',
 '000 apply',
 '000 australia',
 '000 boys']
# for doc = 0
doc = 0
feature_index = tfidf_matrix[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
for w, s in [(features[i], s) for (i, s) in tfidf_scores]: # size = 55
    print(w, s)

learns aside 0.13497681053459026
eventually learns 0.13497681053459026
duo eventually 0.13497681053459026
owner duo 0.13497681053459026
woody owner 0.13497681053459026
buzz woody 0.13497681053459026
separate buzz 0.13497681053459026
circumstances separate 0.13497681053459026
buzz circumstances 0.13497681053459026
plots buzz 0.13497681053459026
woody plots 0.13497681053459026
heart woody 0.13497681053459026
andy heart 0.13497681053459026
place andy 0.13497681053459026
losing place 0.13497681053459026

#### other way of displaying
# stops = set(stopwords.words("english"))
# wl = nltk.WordNetLemmatizer()

# def clean_text(text):
#     """
#       - Remove Punctuations
#       - Tokenization
#       - Remove Stopwords
#       - stemming/lemmatizing
#     """
#     text_nopunct = "".join([char for char in text if char not in string.punctuation])
#     tokens = re.split("\W+", text)
#     text = [word for word in tokens if word not in stops]
#     text = [wl.lemmatize(word) for word in text]
#     return text

# def extract_topn_from_vector(feature_names, sorted_items, topn=5):
#     """
#       get the feature names and tf-idf score of top n items in the doc,                 
#       in descending order of scores. 
#     """

#     # use only top n items from vector.
#     sorted_items = sorted_items[:topn]

#     results= {} 
#     # word index and corresponding tf-idf score
#     for idx, score in sorted_items:
#         results[feature_names[idx]] = round(score, 3)

#     # return a sorted list of tuples with feature name and tf-idf score as its element(in descending order of tf-idf scores).
#     return sorted(results.items(), key=lambda kv: kv[1], reverse=True)

# count_vect = CountVectorizer(analyzer=clean_text, tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)                                        
# freq_term_matrix = count_vect.fit_transform(data['text_body'])

# tfidf = TfidfTransformer(norm="l2")
# tfidf.fit(freq_term_matrix)  

# feature_names = count_vect.get_feature_names()

# # sample document
# doc = 'watched horrid thing TV. Needless say one movies watch see much worse get.'

# tf_idf_vector = tfidf.transform(count_vect.transform([doc]))

# coo_matrix = tf_idf_vector.tocoo()
# tuples = zip(coo_matrix.col, coo_matrix.data)
# sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# # extract only the top n elements.
# # Here, n is 10.
# word_tfidf = extract_topn_from_vector(feature_names,sorted_items,10)

# print("{}  {}".format("features", "tfidf"))  
# for k in word_tfidf:
#     print("{} - {}".format(k[0], k[1])) 

# features  tfidf
# Needless - 0.515
# horrid - 0.501
# worse - 0.312
# watched - 0.275
# TV - 0.272
# say - 0.202
# watch - 0.199
# thing - 0.189
# much - 0.177
# see - 0.164   


Metadata Based Recommende
==============================
x # (55, 1)
count_matrix[0,:].nonzero() # (2, 32)

Popularity and Ratings
========================
# indices
title
Toy Story                      0
Jumanji                        1
Grumpier Old Men               2
Waiting to Exhale              3
Father of the Bride Part II    4
dtype: int64

Collaborative Filtering
=========================
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

Hybrid Recommender
======================
# id_map ->  Index(['movieId', 'id'], dtype='object')
 	                              movieId 	    id
title 		

Toy Story 	                         1 	         862.0
Jumanji 	                         2 	         8844.0
Grumpier Old Men 	                 3 	         15602.0
Waiting to Exhale 	                 4 	         31357.0
Father of the Bride Part II 	     5 	         11862.0

# indices_map -> Index(['movieId'], dtype='object')

# apply & assign to multiple row & column

#    Field_1  Field_2  Field_3
# a      1.5      2.5     10.0
# b      2.0      4.5      5.0
# c      2.5      5.2      8.0
# d      4.5      5.8      4.8
# e      4.0      6.3     70.0
# f      4.1      6.4      9.0
# g      5.1      2.3     11.1

df = df.apply(lambda x: np.square(x) if x.name in ['b', 'f'] else x, axis=1)

#    Field_1  Field_2  Field_3
# a     1.50     2.50     10.0
# b     4.00    20.25     25.0 --->
# c     2.50     5.20      8.0
# d     4.50     5.80      4.8
# e     4.00     6.30     70.0
# f    16.81    40.96     81.0 --->
# g     5.10     2.30     11.1


df = df.assign(Product=lambda x: (x['Field_1'] * x['Field_2'] * x['Field_3']))

#       Field_1  Field_2  Field_3  Product
# a     1.50     2.50     10.0     37.5000
# b     4.00    20.25     25.0   2025.0000
# c     2.50     5.20      8.0    104.0000
# d     4.50     5.80      4.8    125.2800
# e     4.00     6.30     70.0   1764.0000
# f    16.81    40.96     81.0  55771.5456
# g     5.10     2.30     11.1    130.2030

