import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from utils.logger import setup_logger
from models.models_structure import PredictDataStructure, Predict2DataStructure


logger = setup_logger()

allowed_creators = [
    'B5500046-D29B-11DF-812D-00151722D3B0',
    '88E7DB79-5738-11DE-A217-001377322DDB',
    '4CC2629A-04F2-11E8-93F7-00155D04D315',
    '6AD59DF0-8C8F-11E7-80BD-1418775F6D11',
    'AA90B897-09E7-11EA-93FD-00155D04D315',
    '1BFA0522-04F2-11E8-93F7-00155D04D315',
    '0D07534D-574E-11E8-93F7-00155D04D315',
    'A9EDA00E-6A1A-11DE-BB11-00151722D3B0'
]

# load model 1
try:
    regressor = CatBoostRegressor()
    regressor.load_model('models/model_v_0.2.cbm')
except Exception:
    logger.exception('Exception while loading model v0.2')

# load model 2
try:
    regressor_2 = CatBoostRegressor()
    regressor_2.load_model('models/model_v_1.0.cbm')
except Exception:
    logger.exception('Exception while loading model v1')

app = FastAPI()


@app.post("/predict")
def predict(data: PredictDataStructure):
    logger.info(f'Got predict request: date: {data.date}, product: {data.product_oid}, amount: {data.amount}, price: {data.price}, creator: {data.creator_oid}')

    prepared_data = pd.DataFrame([{
        'Date': data.date,
        'Product': data.product_oid,
        'Amount': data.amount,
        'Price': data.price,
        'Creator': data.creator_oid
    }])

    if data.creator_oid not in allowed_creators:
        logger.warning(f'Creator not allowed: {data.creator_oid}')
        raise HTTPException(status_code=422, detail='creator not allowed')

    try:
        prediction = regressor.predict(prepared_data)[0]
        probable_discount = int(data.price * prediction / 100)

        if probable_discount > 0:
            return {'probable_discount': probable_discount}
        else:
            return {'probable_discount': 0}

    except Exception:
        logger.exception('Exception while predict')
        raise HTTPException(status_code=422, detail='model error')


@app.post("/predict_2")
def predict(data: Predict2DataStructure):
    logger.info(f'Got predict_2 request: date: {data.date}, partner: {data.partner} product: {data.product_oid}, amount: {data.amount}, price: {data.price}, avg_price: {data.avg_price}')

    prepared_data = pd.DataFrame([{
        'Date': data.date,
        'Partner': data.partner,
        'Product': data.product_oid,
        'Amount': data.amount,
        'AvgPrice': data.avg_price,
        'Price': data.price,
    }])

    try:
        prediction = regressor_2.predict(prepared_data)[0]
        probable_discount = int(float(data.price) * prediction / 100)

        if probable_discount > 0:
            return {'probable_discount': probable_discount}
        else:
            return {'probable_discount': 0}

    except Exception:
        logger.exception('Exception while predict')
        raise HTTPException(status_code=422, detail='model error')
