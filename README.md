# 🛡️ Phishing Website Detection Using Machine Learning

## 📌 Overview

Phishing is one of the most dangerous cybersecurity threats where attackers create fraudulent websites to steal sensitive information such as login credentials, banking details, and personal data.

Traditional blacklist-based detection systems fail to detect newly created phishing websites effectively.  
This project uses **Machine Learning** to automatically classify websites as:

- 🔴 Phishing (`-1`)
- 🟢 Legitimate (`1`)

By analyzing URL-based, domain-based, and content-based features, the model provides reliable phishing detection.

---

## 🚨 What is Phishing?

Phishing is a cyber attack where fraudulent websites mimic legitimate platforms to trick users into sharing sensitive information.

### Why is it important?

Phishing can lead to:
- Financial loss  
- Identity theft  
- Data breaches  
- Organizational security risks  

Machine Learning enables scalable and automated phishing detection.

---

## 📊 Dataset Overview

The dataset contains **30 features** extracted from websites.

### Labels:
- `-1` → Phishing  
- `1` → Legitimate  

---

## 🔍 Feature Categories

### 1️⃣ URL-Based Features
- Having IP Address  
- URL Length  
- URL Shortening Service  
- Prefix/Suffix  
- HTTPS usage  

### 2️⃣ Domain-Based Features
- SSL State  
- Domain Registration Length  
- DNS Record  
- Website Traffic  

### 3️⃣ Content-Based Features
- Request URL  
- Links in Tags  
- Iframes  
- Anchor URLs  

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Handling missing values  
- Encoding categorical features  
- Normalizing feature values  
- Splitting dataset into training and testing sets  

---

## 🤖 Model Selection & Training

The following Machine Learning models were tested:

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Gradient Boosting (XGBoost)  

Training was performed using **Scikit-learn**.

---

## 📈 Model Evaluation

Performance metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

### Results

- High overall accuracy achieved  
- Random Forest and XGBoost performed best  
- Strong precision and recall indicate reliable phishing detection  

---

## 🌐 Web Application Deployment

The trained model is deployed using:

- Flask (Backend)  
- HTML/CSS (Frontend)  
- Docker (Containerization)  

---

## 🖥️ Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Nandanbharadwaj3797/PhisingClassifier.git
cd PhisingClassifier
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
python app.py
```

Application runs at:

http://localhost:8080

### 🐳 Run Using Docker

#### Build Docker Image

```bash
docker build -t phishing-classifier .
```

#### Run Container

```bash
docker run -p 5000:5000 phishing-classifier
```

Access the application at:

http://localhost:5000

---

## 🔄 Environment Configuration

The application supports dynamic port configuration:

```python
port = int(os.environ.get("PORT", 8080))
```

- Default (Local): 8080
- Docker: 5000

---

## 🏗️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- Docker
- HTML/CSS
