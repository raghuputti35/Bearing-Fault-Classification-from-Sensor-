# ⚙️ Bearing Fault Classification (SVM, Random Forest, KNN)

Classify bearing faults from sensor data using **scikit-learn** and **Streamlit**.

- `train_model.py` → trains SVM, Random Forest and KNN on your Kaggle CSV and saves them.
- `app.py` → simple UI only. Enter sensor values by hand, or upload a CSV, and get the predicted fault type. No training and no graphs in the app.

## Project structure

```
bearing_fault_project/
├── train_model.py    # training script (run once)
├── app.py            # Streamlit UI (prediction only)
├── requirements.txt
├── README.md
├── data/             # put your Kaggle CSV here
└── models/           # trained model is saved here
```

## Step 1 — Install (Python 3.9+)

```bash
cd bearing_fault_project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

## Step 2 — Get a dataset from Kaggle

No data is included. Download a real one:

1. Search <https://www.kaggle.com/datasets> for **"bearing fault"** or **"bearing fault diagnosis"**.
2. Pick a dataset with a **CSV** where each row is one sample, the columns are numeric
   sensor values/features (RMS, kurtosis, skewness, ...), and one column is the fault label
   (Normal, Inner Race, Outer Race, ...).
3. Download, unzip, and copy the CSV into the **`data/`** folder.

Or with the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d <owner>/<dataset-name> -p data --unzip
```

> If the labels are in a separate file, merge them into one CSV first. If the dataset only has
> raw signals, extract features (mean, std, RMS, kurtosis) per window first.

## Step 3 — Train

```bash
python train_model.py
```

By default it uses the first CSV in `data/` and treats the **last column** as the fault label.
If your file is different:

```bash
python train_model.py --data data/myfile.csv --label fault_type
```

It prints accuracy and a classification report for each model and saves
`models/bearing_models.joblib`.

## Step 4 — Run the app

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>.

- **Enter values manually:** fill in each sensor value → click **Predict**.
- **Upload a CSV file:** the CSV must contain the same feature columns used in training
  (the app lists them). You get a table of predictions and a download button.
- Use the sidebar to choose SVM, Random Forest or KNN.

## Troubleshooting

- **"No trained model found"** → run `python train_model.py` first.
- **"No CSV found"** → the dataset is not in `data/`.
- **"Missing columns"** in the app → your uploaded CSV needs the same feature column names as the training data.
- **Changed dataset?** Run `python train_model.py` again.
