# 🏥 ArogyaGram – AI-Powered Rural Healthcare Assistance System

ArogyaGram is an AI-driven healthcare support system designed to assist individuals—especially those in rural and underserved areas—in understanding potential health conditions based on their symptoms. The project focuses on providing early-stage medical guidance in a safe, explainable, and user-friendly manner.

The system does not aim to replace doctors or professional diagnosis. Instead, it acts as an **intelligent pre-screening tool** that helps users recognize possible risks and take timely action.

---

## 🌍 Motivation and Problem Context

Access to healthcare remains a major challenge in rural regions. Many individuals either delay seeking medical attention or rely on unreliable sources of information. This often leads to worsening conditions that could have been prevented with early intervention.

ArogyaGram was developed with the idea that even a basic, reliable AI system can:

* Help users interpret their symptoms
* Encourage timely medical consultation
* Reduce panic caused by misinformation
* Provide simple and understandable guidance

The goal is to build a system that is **technically strong but also socially responsible and accessible**.

---

## 🧠 System Overview

At its core, ArogyaGram takes **symptoms as input** and produces a **structured medical guidance output**.

A typical interaction looks like:

User input:

```
fever, headache, nausea
```

System output:

* Top possible diseases
* Confidence (probability)
* Severity level (mild, moderate, emergency)
* Precautionary advice
* Suggestions for diagnostic tests (if applicable)
* Warning messages in case of low confidence or critical conditions

The system is designed to balance **accuracy with safety**, ensuring that users are not misled or unnecessarily alarmed.

---

## ⚙️ Technology Stack

The project combines modern NLP techniques with practical web deployment tools.

**Programming Language**

* Python

**Machine Learning & NLP**

* PyTorch
* HuggingFace Transformers
* DistilBERT (core model)

**Data Processing**

* Pandas
* NumPy
* Scikit-learn

**Web Application**

* Flask

**Development Tools**

* Git & GitHub
* PowerShell
* VS Code

---

## 📊 Dataset and Data Handling

The model is trained on a **public symptom–disease dataset**, commonly used for healthcare machine learning experiments.

Each entry in the dataset represents a disease and its associated symptoms.

### Example:

```
itching, skin rash, nodal skin eruptions → Fungal infection
```

### Data Files Used

The project uses multiple structured files:

* `clean_dataset.csv` → cleaned dataset
* `clean_dataset_dedup.csv` → duplicate-free dataset
* `train.csv`, `val.csv`, `test.csv` → split datasets
* `precaution.csv` → disease precautions
* `symptom_Description.csv` → symptom explanations

---

## 🧹 Data Preprocessing

Before training the model, the dataset was carefully processed to improve quality and consistency.

Key steps included:

* Removing duplicate entries to prevent bias
* Converting structured symptom columns into natural language text
* Normalizing symptom representations
* Splitting data into training, validation, and test sets
* Ensuring no overlap between splits

### Example Transformation

From structured format:

```
Symptom_1: itching  
Symptom_2: skin rash  
Symptom_3: nodal eruptions  
```

To natural language:

```
"itching, skin rash, nodal eruptions"
```

This step was critical to enable the use of transformer-based NLP models.

---

## 🧠 Model Architecture

The system uses **DistilBERT**, a lightweight transformer model optimized for efficiency.

### Why DistilBERT?

* Faster than traditional BERT
* Lower memory usage
* Suitable for real-time applications
* Performs well on classification tasks

The model is fine-tuned for **multi-class classification**, predicting among approximately **41 diseases**.

---

## 🏋️ Model Training

Training is performed using:

```
train_transformer_clean.py
```

The training pipeline includes:

* Tokenization of input text
* Feeding tokens into DistilBERT
* Fine-tuning classification head
* Backpropagation using gradient descent
* Monitoring loss and performance

Training logs include:

* Loss values
* Learning rate
* Gradient norms
* Epoch progression

---

## 📈 Model Evaluation

Evaluation is performed using:

```
eval.py
```

Metrics used:

* Top-1 Accuracy
* Top-3 Accuracy
* Macro F1 Score
* Class-wise recall

Results showed strong performance on the dataset, though real-world generalization is still an area for improvement.

---

## 🔍 Prediction System

Predictions are handled through:

```
predict_clean_plus.py
```

Unlike a basic classifier, ArogyaGram produces **multi-layered outputs**:

* Ranked disease predictions
* Probability scores
* Severity assessment
* Precautionary advice
* Symptom explanations
* Confidence warnings

---

## 🛡️ Safety and Responsibility Layer

One of the most important aspects of this project is the **safety layer**, which ensures responsible output.

This includes:

* Detection of critical diseases (e.g., heart attack)
* Emergency-level alerts
* Low-confidence warnings
* Handling of unknown symptoms
* Suggestion of diagnostic tests for confirmation

This layer ensures the system does not:

* Mislead users
* Create unnecessary panic
* Provide unsafe guidance

---

## 🌐 Web Interface

The system is deployed using Flask for ease of use.

To run the application:

```
python app_flask.py
```

Then open:

```
http://127.0.0.1:5000
```

Users can input symptoms and receive results instantly through a simple and clean interface.

---

## 🗂️ Project Structure

```
rural-health-proto/
├── app_flask.py
├── predict_clean_plus.py
├── train_transformer_clean.py
├── eval.py
├── data/
├── trained_model_clean/
├── reports/
└── venv/
```

---

## 🔮 Future Enhancements

The project is designed to evolve further. Planned improvements include:

* Hindi language support for rural accessibility
* Voice-based symptom input
* Image-based disease detection
* Larger and more diverse datasets
* Improved model calibration
* Mobile-friendly UI

---

## ⚠️ Disclaimer

This system is **not a medical diagnosis tool**.

It is intended for educational and research purposes only.
Users should always consult a qualified healthcare professional for medical advice.

---

## 👩‍💻 Author

Vinshee Kulshreshtha
Artificial Intelligence / Machine Learning Enthusiast

---

## 🌱 Final Note

ArogyaGram represents an effort to combine:

* AI technology
* healthcare awareness
* ethical system design

to build something that is not just technically correct, but also **meaningful and responsible**.

---

# 🚀 Next step (do this)

1. Replace your current README with this
2. Run:

git add README.md
git commit -m "Improved detailed README"
git push
