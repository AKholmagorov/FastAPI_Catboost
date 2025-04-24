import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.logger import setup_logger


class PredictDataStructure(BaseModel):
    date: str
    amount: str
    price: float
    creator: str


logger = setup_logger()

allowed_creators = [
    '7CF68F9A-D673-11EC-940C-00155D04D315',
    '86AEFC14-2294-11DE-A1EB-001377322DDB',
    '6AD59DF0-8C8F-11E7-80BD-1418775F6D11',
    'B5500046-D29B-11DF-812D-00151722D3B0',
    '88E7DB79-5738-11DE-A217-001377322DDB',
    '614E9AE0-C765-11EA-9407-00155D04D315',
    '6A3FE959-711B-11EB-9409-00155D04D315',
    'A9EDA00E-6A1A-11DE-BB11-00151722D3B0',
    'AA90B897-09E7-11EA-93FD-00155D04D315',
    'C283DF61-C765-11EA-9407-00155D04D315',
    '88E7DB85-5738-11DE-A217-001377322DDB',
    '1BFA0522-04F2-11E8-93F7-00155D04D315',
    '0E925BFC-C766-11EA-9407-00155D04D315',
    'B7810962-6ABA-11E7-93F1-00155D04D315'
]

try:
    regressor = CatBoostRegressor()
    regressor.load_model('models/model.cbm')
except Exception:
    logger.exception('Exception while loading model')


app = FastAPI()


@app.post("/predict")
def predict(data: PredictDataStructure):
    logger.info(f'Got predict request: date: {data.date}, amount: {data.amount}, price: {data.price}, creator: {data.creator}')

    prepared_data = pd.DataFrame([{
        'Date': data.date,
        'Amount': data.amount,
        'Price': data.price,
        'Creator': data.creator
    }])

    if data.creator not in allowed_creators:
        logger.warning(f'Creator not allowed: {data.creator}')
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
