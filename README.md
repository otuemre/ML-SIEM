
# Machine Learning SIEM (Security Information and Event Management)

A project exploring unsupervised anomaly detection for identifying account takeover attempts in login data using AutoEncoders. Built using the [RBA Dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset) and structured for future integration into a full ML-based SIEM system.

---

## Dataset

- **Source:** [Kaggle – RBA Dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset)
- **Size:** ~32 million login attempts
- **Features Used:**
  - Country
  - Device type
  - Login hour & weekday
  - Browser name
  - OS name
  - IP address (split into 4 octets)
  - Login success indicator
  - Attack IP flag
- **Target (only used for evaluation):** `is_account_takeover`

---

## Project Pipeline

1. **Data Preprocessing**
   - Removed high-cardinality and redundant features (e.g., user ID, ASN, raw IP string)
   - Extracted hour/day from timestamp
   - Scaled all features using `MinMaxScaler`
   - Converted IP address into 4 normalized octets (e.g., `ip_1` to `ip_4`)
   - Removed over 24.5 million duplicate rows

2. **Model Architecture**
   - Fully connected AutoEncoder
   - Batch Normalization for stability
   - Sigmoid output activation for MinMax-scaled data
   - Trained on only normal login data (unsupervised)

3. **Evaluation**
   - Used reconstruction error (MSE) to flag anomalies
   - Threshold tuned using reconstruction error percentile
   - Tested on full dataset with true labels (`is_account_takeover`)

---

## Key Findings

| Configuration   | Recall  | Precision | Notes                                   |
|-----------------|---------|-----------|-----------------------------------------|
| Vanilla AE      | ~2%     | ~0%       | Couldn't separate anomalies             |
| AE + IP Octets  | **50%** | ~0%       | Massive gain with IP as behavior signal |
| AE + Duplicates | ~22%    | ~0%       | Helped generalization slightly          |

**IP octet extraction enabled AutoEncoder to catch anomalies by revealing subtle behavior patterns**

Without user IDs or behavioral history, anomaly detection remains limited in precision

---

## Folder Structure

```
project-root/
│
├── data/                  # Preprocessed and scaled data
├── models/                # Saved AutoEncoder models
├── notebooks/             # Jupyter notebooks for EDA, training, and evaluation
├── src/                   # Future: Python pipeline scripts
├── README.md              # You are here
```

---

## Future Work

- [ ] Integrate **Denoising AutoEncoder** for better generalization
- [ ] Build a **model pipeline** for real-time inference
- [ ] Test TensorFlow Hub-based models for tabular anomaly detection
- [ ] Explore session-based or user ID–aware behavior modeling (if identifiers become available)

---

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- Dask (for scalable preprocessing)
- Pandas / NumPy / Seaborn / Matplotlib
- Scikit-learn (for metrics)
- Jupyter Notebooks

---

## Author

**Emre Otu**  
📍 Computer Science & Artificial Intelligence Student  
🔗 [LinkedIn](https://www.linkedin.com/in/emreotu)  
📂 [GitHub](https://github.com/otuemre)

---