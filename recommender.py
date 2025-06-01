import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

url_data="http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
url_item="http://files.grouplens.org/datasets/movielens/ml-100k/u.item"
column_names=["user_id","item_id","rating","timestamp"]
df=pd.read_csv(url_data, sep="\t" ,names=column_names)
movie_columns=[
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Thriller",
    "War",
    "Western",
]
df_movies = pd.read_csv(url_item, sep="|", names=movie_columns, encoding="latin-1")
movie_titles=dict(zip(df_movies["item_id"],df_movies["title"]))
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
trainset, testset=train_test_split(data,test_size=0.2)
model=SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions)
def recommend(user_id,num_recommendations=5):
    all_items=df["item_id"].unique()
    predict_rating=[model.predict(user_id,item_id).est for item_id in all_items]
    item_ratings=list(zip(all_items,predict_rating))
    item_ratings.sort(key=lambda x: x[1], reverse=True)
    top_items=item_ratings[:num_recommendations]
    top_items_with_titles = [
        (movie_titles.get(item_id, f"Movie ID {item_id} "), rating)
        for item_id, rating in top_items
    ]
    return top_items_with_titles

user_id=196
recommendations=recommend(user_id,5)
print("Top 5 recommendations for user {}:".format(user_id))
for title, rating in recommendations:
    print(f"{title}: Predicted Rating: {rating:.2f}")
