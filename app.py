from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load XGBoost model
with open("best_xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

# Custom label encoding sesuai training
neighborhood_mapping = {'MeadowV': 0,
 'IDOTRR': 1,
 'BrDale': 2,
 'BrkSide': 3,
 'Edwards': 4,
 'OldTown': 5,
 'Sawyer': 6,
 'Blueste': 7,
 'SWISU': 8,
 'NPkVill': 9,
 'NAmes': 10,
 'Mitchel': 11,
 'SawyerW': 12,
 'NWAmes': 13,
 'Gilbert': 14,
 'Blmngtn': 15,
 'CollgCr': 16,
 'Crawfor': 17,
 'ClearCr': 18,
 'Somerst': 19,
 'Veenker': 20,
 'Timber': 21,
 'StoneBr': 22,
 'NridgHt': 23,
 'NoRidge': 24}

qual_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
garage_finish_mapping = {'Unf': 1, 'RFn': 2, 'Fin': 3, 'None': 0}

@app.route('/')
def form():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        data = {
            'Neighborhood': neighborhood_mapping[form_data['Neighborhood']],
            'ExterQual': qual_mapping[form_data['ExterQual']],
            'BsmtQual': qual_mapping[form_data['BsmtQual']],
            'KitchenQual': qual_mapping[form_data['KitchenQual']],
            'GarageFinish': garage_finish_mapping[form_data['GarageFinish']],
            'OverallQual': int(form_data['OverallQual']),
            'GrLivArea': int(form_data['GrLivArea']),
            'GarageCars': int(form_data['GarageCars']),
            'YearBuilt': int(form_data['YearBuilt']),
            'GarageArea': int(form_data['GarageArea']),
            'FullBath': int(form_data['FullBath']),
            'TotalBsmtSF': int(form_data['TotalBsmtSF'])
        }

        df = pd.DataFrame([data])  # Model expects DataFrame
        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

        return f"<h2>Prediksi Harga Rumah: ${prediction:,.2f}</h2><a href='/'>Kembali</a>"
    
    except Exception as e:
        return f"<h2>Error saat prediksi:</h2><p>{str(e)}</p><a href='/'>Kembali</a>"

if __name__ == '__main__':
    app.run(debug=True)
