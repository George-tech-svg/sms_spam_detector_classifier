# SMS Spam Classifier - Production Web Application with Batch Testing
import streamlit as st
import joblib
import re
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data on first run
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('models/final_spam_classifier.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

# Text preprocessing function
def preprocess_message(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Prediction function for single message
def predict_message(message, model, vectorizer):
    processed = preprocess_message(message)
    vectorized = vectorizer.transform([processed])
    vectorized_dense = vectorized.toarray()
    prediction = model.predict(vectorized_dense)[0]
    
    if prediction == 1:
        return "SPAM"
    else:
        return "NOT SPAM"

# Batch prediction function
def predict_batch(messages, model, vectorizer):
    results = []
    for msg in messages:
        if msg.strip():
            prediction = predict_message(msg, model, vectorizer)
            results.append({
                'message': msg,
                'prediction': prediction,
                'length': len(msg)
            })
    return pd.DataFrame(results)

# Color function for dataframe
def color_prediction(val):
    if val == 'SPAM':
        return 'color: red'
    else:
        return 'color: green'

# Main app
def main():
    st.set_page_config(
        page_title="SMS Spam Detector",
        page_icon="",
        layout="wide"
    )
    
    st.title("SMS Spam Detector")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("Detects whether SMS messages are spam or not.")
        st.write("**Model:** SVM")
        st.write("**Accuracy:** 97.94%")
        st.write("**Precision:** 97.01%")
        st.write("**Recall:** 87.25%")
        st.markdown("---")
        st.header("Sample Messages")
        
        sample_spam = "CONGRATULATIONS! You won $1000. Call now"
        sample_ham = "Hey, see you at the meeting tomorrow"
        
        if st.button("Add Spam Sample"):
            st.session_state.sample = sample_spam
        if st.button("Add Ham Sample"):
            st.session_state.sample = sample_ham
    
    # Tab layout
    tab1, tab2 = st.tabs(["Single Message", "Batch Testing"])
    
    # TAB 1: Single Message Testing
    with tab1:
        st.subheader("Test One Message")
        
        default_text = st.session_state.get('sample', '')
        user_input = st.text_area("Enter your message:", value=default_text, height=100)
        
        if st.button("Predict Single", type="primary"):
            if user_input and user_input.strip():
                with st.spinner("Analyzing..."):
                    download_nltk_data()
                    model, vectorizer = load_models()
                    result = predict_message(user_input, model, vectorizer)
                    
                    st.markdown("---")
                    st.subheader("Result:")
                    
                    if result == "SPAM":
                        st.error(f"**{result}**")
                    else:
                        st.success(f"**{result}**")
            else:
                st.warning("Please enter a message")
    
    # TAB 2: Batch Testing (Multiple Messages)
    with tab2:
        st.subheader("Test Multiple Messages at Once")
        
        st.markdown("Enter multiple messages, one per line:")
        
        batch_input = st.text_area(
            "Messages (one per line):",
            height=200,
            placeholder="Example:"
            "\nWINNER! You won $1000"
            "\nHey, how are you?"
            "\nClaim your prize now"
            "\nSee you tomorrow at 5pm"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Batch Test"):
                if batch_input and batch_input.strip():
                    with st.spinner("Analyzing batch..."):
                        download_nltk_data()
                        model, vectorizer = load_models()
                        
                        # Split by new line
                        messages = batch_input.strip().split('\n')
                        
                        # Get predictions
                        results_df = predict_batch(messages, model, vectorizer)
                        
                        # Store in session state
                        st.session_state.results_df = results_df
                else:
                    st.warning("Please enter at least one message")
        
        with col2:
            if st.button("Clear Batch"):
                st.session_state.results_df = None
                st.rerun()
        
        # Display results if available
        if st.session_state.get('results_df') is not None:
            results_df = st.session_state.results_df
            
            st.markdown("---")
            st.subheader("Batch Results")
            
            # Summary metrics
            total = len(results_df)
            spam_count = len(results_df[results_df['prediction'] == 'SPAM'])
            ham_count = len(results_df[results_df['prediction'] == 'NOT SPAM'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", total)
            with col2:
                st.metric("Spam Detected", spam_count)
            with col3:
                st.metric("Not Spam", ham_count)
            
            # Display results table with color coding
            styled_df = results_df.style.map(color_prediction, subset=['prediction'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Create visualization chart
            st.subheader("Results Visualization")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['green', 'red']
            counts = [ham_count, spam_count]
            labels = ['Not Spam', 'Spam']
            
            bars = ax.bar(labels, counts, color=colors, edgecolor='black')
            ax.set_ylabel('Number of Messages')
            ax.set_title('Batch Test Results: Spam vs Not Spam')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', fontweight='bold')
            
            st.pyplot(fig)
            
            # Download results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="spam_detection_results.csv",
                mime="text/csv"
            )

# Run the app
if __name__ == "__main__":
    main()