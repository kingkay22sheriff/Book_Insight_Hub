import numpy as np
import pickle

# Load artifacts
model = pickle.load(open('model.pkl', 'rb'))
books_name = pickle.load(open('books_name.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

# Recommendation function
def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        print(f"Recommendations for '{book_name}':")
        for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                print(j)
    except IndexError:
        print(f"Book '{book_name}' not found in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
    recommend_book(book_name)