# ADR2Condition 🧪

**ADR2Condition** is a semi-built machine learning project designed to predict possible medical conditions based on adverse drug reactions (ADRs) or side effects. It also displays useful information such as the average condition rating and the number of user reviews.

> ⚠️ This project is a prototype and not intended for medical use. It was built using limited data for research and educational purposes.

---

## 🚀 Features

- Predicts medical condition based on inputted side effects.
- Returns rating and number of reviews for the predicted condition.
- Model trained using TF-IDF vectorization and Naive Bayes classification.
- Deployed locally using **Streamlit** for interactive usage.

---

## 🧠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle (for model persistence)

---

## 📁 Project Structure
.
├── main.py # Trains and saves the ML model
├── model.pkl # Trained model file
├── vectorizer.pkl # TF-IDF vectorizer file
├── app.py # Streamlit UI for predictions
├── sample_data.csv # Sample dataset
├── requirements.txt # Dependencies
├── README.md # This file


---

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/adr2condition.git
cd adr2condition
```

2. Install dependencies
```bash
Copy
Edit
pip install -r requirements.txt

```
3. Run Streamlit App
```bash
streamlit run app.py
```
4.🧪 Example Usage
```bash
Enter side effects like:
hives, nausea, slurred speech

Output:
Predicted Condition: Acne
Rating: 6.5
Reviews: 230

```
## 📌 Note
This project is a semi-built prototype due to the lack of comprehensive ADR-to-condition datasets. It currently works on sample data and demonstrates the feasibility of such a model.

