import numpy as np
import pickle
import os
from pathlib import Path

class BookRecommender:
    def __init__(self):
        
        self.model = None
        self.books_name = None
        self.final_rating = None
        self.book_pivot = None
        self.load_artifacts()

    def load_artifacts(self):
        
        try:
            base_path = Path(__file__).parent
            self.model = pickle.load(open(base_path / 'model.pkl', 'rb'))
            self.books_name = pickle.load(open(base_path / 'books_name.pkl', 'rb'))
            self.final_rating = pickle.load(open(base_path / 'final_rating.pkl', 'rb'))
            self.book_pivot = pickle.load(open(base_path / 'book_pivot.pkl', 'rb'))
            print("All artifacts loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: Missing file - {e.filename}")
            print(f"Please ensure all required pickle files are in: {base_path}")
            raise
        except Exception as e:
            print(f"Error loading artifacts: {str(e)}")
            raise

    def recommend_book(self, book_name, num_recommendations=5):
        try:
            # Verify book exists in dataset
            if book_name not in self.book_pivot.index:
                return f"Error: Book '{book_name}' not found in the dataset."

            # Get book ID and generate recommendations
            book_id = np.where(self.book_pivot.index == book_name)[0][0]
            distance, suggestion = self.model.kneighbors(
                self.book_pivot.iloc[book_id, :].values.reshape(1, -1), 
                n_neighbors=num_recommendations + 1  # +1 because input book will be removed
            )

            # Extract recommended books
            recommendations = []
            for i in range(len(suggestion)):
                books = self.book_pivot.index[suggestion[i]]
                recommendations.extend(books)

            # Remove the input book from recommendations if present
            recommendations = [book for book in recommendations if book != book_name]

            return recommendations[:num_recommendations]

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def get_available_books(self):
        return list(self.book_pivot.index)

    def get_book_details(self, book_name):
        try:
            book_idx = np.where(self.final_rating['Title'] == book_name)[0][0]
            book_data = self.final_rating.iloc[book_idx]
            return {
                'title': book_data['Title'],
                'img_url': book_data['Img_url'] if 'Img_url' in book_data else None
            }
        except:
            return None


def main():
    try:
        print("Initializing Book Recommender System...")
        recommender = BookRecommender()
        
        book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
        
        print(f"\nGenerating recommendations for: {book_name}")
        recommendations = recommender.recommend_book(book_name)
        
        if isinstance(recommendations, list):
            print("\nRecommended Books:")
            for i, book in enumerate(recommendations, 1):
                book_details = recommender.get_book_details(book)
                if book_details:
                    print(f"{i}. {book_details['title']}")
                else:
                    print(f"{i}. {book}")
        else:
            print(recommendations)
    except Exception as e:
        print(f"Program failed: {str(e)}")
        print("\nPlease ensure all required pickle files are present in the same directory:")
        print("- model.pkl")
        print("- books_name.pkl")
        print("- final_rating.pkl")
        print("- book_pivot.pkl")

if __name__ == "__main__":
    main()