from flask import Flask
from flask import request
import pickle
import pandas as pd
import json

app = Flask(__name__)

# import model
model_pkl = open('house_pricing_ridge', 'rb')
lin = pickle.load(model_pkl)
# import fit
fit_pkl = open('house_pricing_fit', 'rb')
scaler = pickle.load(fit_pkl)

# import means to fill missing value
means_json = open('means.json', 'rb')
means = json.load(means_json)

@app.route('/unique_pred')
def get_prediction():
    X = pd.DataFrame(columns=['CRIM','ZN','INDUS','CHAS','RM','RAD','PTRATIO','B','LSTAT'])
    # populate with params values
    X.loc[0,'CRIM']=request.args.get('CRIM')
    X.loc[0, 'ZN']=request.args.get('ZN')
    X.loc[0, 'INDUS']=request.args.get('INDUS')
    X.loc[0, 'CHAS']=request.args.get('CHAS')
    X.loc[0, 'RM']=request.args.get('RM')
    X.loc[0,'RAD']=request.args.get('RAD')
    X.loc[0, 'PTRATIO']=request.args.get('PTRATIO')
    X.loc[0, 'B']=request.args.get('B')
    X.loc[0, 'LSTAT']=request.args.get('LSTAT')
    # if missing values, fill with X_train values
    for col in X.columns:
        X[col].fillna(means[col], inplace=True)
    # normalized input
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    # predict
    result = lin.predict(X)

    return 'the prediction for your point is {}'.format(str(round(result[0],4)))


@app.route("/mul_pred", methods=["POST"])
def get_multiple_predictions():
    # read json
    param_dict = request.get_json()
    # convert json to dict
    param_dict = json.loads(param_dict)
    X = pd.DataFrame(columns=['CRIM','ZN','INDUS','CHAS','RM','RAD','PTRATIO','B','LSTAT'])
    for sample in range(len(param_dict["CRIM"])):
        for col in X.columns:
            # populate the df
            try:
                # if we got value via json
                X.loc[sample,col]=param_dict[col][sample]
            except IndexError:
                # if we did not get value via json, fill with mean value
                X.loc[sample,col]=means[col]
    # normalized the df
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    # predict
    prediction = lin.predict(X)
    # convert to json
    pred_json = json.dumps(prediction.tolist())
    return pred_json


if __name__ == '__main__':
    app.run()