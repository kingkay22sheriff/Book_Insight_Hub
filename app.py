import streamlit as st
import numpy as np
import os
from pathlib import Path


try:
    import sklearn
except ImportError:
    st.error("Missing required package 'scikit-learn'. Please install it using: pip install scikit-learn")
    st.stop()

try:
    import pickle
except ImportError:
    st.error("Missing required package 'pickle-mixin'. Please install it using: pip install pickle-mixin")
    st.stop()


st.set_page_config(
    page_title="Book Recommender",
    layout="wide"
)

# Define the data loading function
@st.cache_resource
def load_model_files():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model = pickle.load(open(os.path.join(current_dir, 'model.pkl'), 'rb'))
        books_name = pickle.load(open(os.path.join(current_dir, 'books_name.pkl'), 'rb'))
        final_rating = pickle.load(open(os.path.join(current_dir, 'final_rating.pkl'), 'rb'))
        book_pivot = pickle.load(open(os.path.join(current_dir, 'book_pivot.pkl'), 'rb'))
        
        return model, books_name, final_rating, book_pivot
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error(f"Current directory: {current_dir}")
        st.error(f"Files in directory: {os.listdir(current_dir)}")
        return None, None, None, None

def fetch_poster(suggestion, book_pivot, final_rating):
    books_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        books_name.append(book_pivot.index[book_id])

    for name in books_name[0]:
        ids = np.where(final_rating['Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['Img_url']
        if isinstance(url, str):
            poster_url.append(url)

    return poster_url

def recommend_books(book_name, model, book_pivot, final_rating):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)

    poster_url = fetch_poster(suggestions, book_pivot, final_rating)
    
    for i in range(len(suggestions)):
        books = book_pivot.index[suggestions[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

def main():
    st.title("Book Recommendation System")
    
    # Load the model files
    with st.spinner("Loading model files..."):
        model, books_name, final_rating, book_pivot = load_model_files()
    
    # Check if model loading was successful
    if model is None or books_name is None or final_rating is None or book_pivot is None:
        st.error("Failed to load required model files. Please check if all files exist in the correct location.")
        st.stop()
        return
    
    
    selected_book = st.selectbox(
        "Type or select a book you like:",
        books_name
    )

    if st.button('Show Recommendations'):
        with st.spinner("Generating recommendations..."):
            try:
                recommended_books, poster_urls = recommend_books(
                    selected_book, 
                    model, 
                    book_pivot, 
                    final_rating
                )

                
                cols = st.columns(5)
                for idx, col in enumerate(cols, 1):
                    with col:
                        st.text(recommended_books[idx])
                        try:
                            st.image(poster_urls[idx])
                        except Exception as e:
                            st.error("Error loading image")
                            
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()