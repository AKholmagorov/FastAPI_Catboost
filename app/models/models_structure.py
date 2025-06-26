from pydantic import BaseModel


class PredictDataStructure(BaseModel):
    date: str
    product_oid: str
    amount: str
    price: float
    creator_oid: str


class Predict2DataStructure(BaseModel):
    date: str
    partner: str
    product_oid: str
    amount: str
    avg_price: str
    price: str