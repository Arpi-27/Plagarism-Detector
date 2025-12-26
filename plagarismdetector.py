
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download NLTK resources (ensure stopwords are downloaded)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to compute plagiarism percentage
def calculate_plagiarism(text1, text2):
    # Check if the input texts are non-empty
    if not text1 or not text2:
        raise ValueError("Both input texts must be non-empty")

    # Preprocess both texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Handle case where after preprocessing the text is empty
    if not text1 or not text2:
        return 0.0  # No similarity if either text is empty after preprocessing

    # Create the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform both texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Return the cosine similarity as a plagiarism percentage
    return cosine_sim[0][0] * 100

# Take user input
print("Enter the original text:")
text1 = input()  # Take the first text input

print("Enter the suspected plagiarized text:")
text2 = input()  # Take the second text input

try:
    plagiarism_percentage = calculate_plagiarism(text1, text2)
    print(f"Plagiarism Percentage: {plagiarism_percentage:.2f}%")
except ValueError as e:
    print(e)
