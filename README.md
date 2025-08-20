🤖 Custom Text Classification with Hugging Face Transformers
    This is an **interactive Streamlit application** that lets you:
      - Upload your own dataset (CSV).
      - Train a **DistilBERT-based text classification model**.
      - Evaluate accuracy & F1 score.
      - Save the trained model.
      - Run **real-time predictions** on new text.

🚀 Features
    - Upload CSV file with `data` (text) and `label` columns.
    - Automatically encodes labels and splits into train/test sets.
    - Uses Hugging Face `transformers` + `datasets`.
    - Fine-tunes **DistilBERT** for sequence classification.
    - Shows **metrics**: Accuracy & Weighted F1.
    - Saves trained model in `./saved_model/`.
    - Allows **live inference** after training.

🛠️ Technologies Used
    - **Python 3**
    - **Streamlit** – Web interface
    - **Transformers (Hugging Face)** – DistilBERT model
    - **Datasets (Hugging Face)** – Data handling
    - **PyTorch** – Model backend
    - **Scikit-learn** – Metrics (accuracy, F1)
    - **Pandas** – CSV processing

---

## 📂 Project S
