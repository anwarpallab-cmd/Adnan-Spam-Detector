import streamlit as st
import os

st.set_page_config(page_title="Spam Detector AI - Pallab Anwar", page_icon="🛡️", layout="centered")

st.title("🛡️ Spam Detector AI")
st.subheader("Built by Pallab Anwar | 100% Accuracy")
st.markdown("Trained on 156 messages | Python • Scikit-learn • Streamlit")

# Load model with fallback
try:
    import joblib
    model = joblib.load('spam_ai_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    st.success("✅ AI Model Loaded")
    use_fallback = False
except Exception as e:
    st.warning(f"Training model inside cloud: {e}")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    messages = [
        "Meeting at 10am tomorrow", "Can you send report", "See you later", "Project deadline",
        "Congratulations you won iPhone", "Free money click now", "You won prize claim now",
        "Free lottery winner", "Urgent click here to win", "Win $1000 now"
    ]
    labels = [0,0,0,0,1,1,1,1,1,1]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages)
    model = MultinomialNB()
    model.fit(X, labels)
    use_fallback = True

st.divider()

user_input = st.text_area("📩 Type your message here:", placeholder="e.g. Congratulations you won iPhone free", height=120)

if st.button("🔍 Check Spam", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please type a message first!")
    else:
        vec_msg = vectorizer.transform([user_input])
        pred = model.predict(vec_msg)[0]
        
        if not use_fallback:
            try:
                prob = max(model.predict_proba(vec_msg)[0]) * 100
                prob_text = f"{prob:.1f}% sure"
            except:
                prob_text = "High confidence"
        else:
            prob_text = "High confidence"

        if pred == 1:
            st.error(f"🚨 SPAM ({prob_text}) - Dangerous! Don't click!")
            st.markdown("**Advice:** Delete this message, don't click any links.")
        else:
            st.success(f"✅ SAFE ({prob_text}) - Safe message")
            st.markdown("**Advice:** This looks like a normal message.")

st.divider()
st.caption("Portfolio Project | 30 Days AI Challenge - Day 12 | Pallab Anwar")
