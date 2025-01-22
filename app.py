import numpy as np
import streamlit as st
import pickle 

st.header("Books Recommendation System using Machine Learning")
model = pickle.load(open('model.pkl', 'rb'))
books_name = pickle.load(open('books_name.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):
    books_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        books_name.append(book_pivot.index[book_id])


    for name in books_name[0]:
        ids = np.where(final_rating['Title'] == name)[0][0]
        ids_index.append(ids)

    for ids in ids_index:
        url = final_rating.iloc[ids]['Img_url']
        if isinstance(url, str):
            poster_url.append(url)


    return poster_url


def recommend_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list, poster_url

    
    
selected_books = st.selectbox(
    "Type or Select a Book",
    books_name,
    help="Choose a book from the dropdown to get recommendations."
)

if st.button("Recommend"):
    recommend_books, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommend_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommend_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommend_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommend_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommend_books[5])
        st.image(poster_url[5])