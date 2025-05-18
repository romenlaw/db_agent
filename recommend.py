import joblib
import json
import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import List

file_path = './memory/recom'
product_cp_pipeline = joblib.load(f'{file_path}/product_cp_prediction_model.pkl')
product_cp_label_encoder = joblib.load(f'{file_path}/prod_cp_label_encoder.pkl')

product_cnp_pipeline = joblib.load(f'{file_path}/product_cnp_prediction_model.pkl')
product_cnp_label_encoder = joblib.load(f'{file_path}/prod_cnp_label_encoder.pkl')

price_pipeline = joblib.load(f'{file_path}/price_prediction_model.pkl')
price_label_encoder = joblib.load(f'{file_path}/price_label_encoder.pkl')

@tool
def recommend_product_wrapper(cp_cnp: str, mis_division: str, mcc: int, postcode: int, revenue: float) -> List[str]:
    """
    Recomend merchant product(s) based on input criteria
    Args:
        cp_cnp: whether the product should be CP/Card Present or CNP/Card Not Present/eComm
        mis_division: the market division. Valid values are RBS, BB, IB&M
        mcc: MCC / Merchant Category Code
        postcode: postcode of trading address
        revenue: monthly revenue / ternover in AUD

    Returns:
        A list of product codes in order of high-to-low propability for recommendation.
    """
    kwargs = locals()
    return recommend_product(**kwargs)

def recommend_product(cp_cnp, mis_division, mcc, postcode, revenue):
    if cp_cnp=='CP':
        prod_pipeline = product_cp_pipeline
        prod_encoder = product_cp_label_encoder
    else:
        prod_pipeline = product_cnp_pipeline
        prod_encoder = product_cnp_label_encoder

    new_data = pd.DataFrame({
        'MIS_DIVISION': [mis_division],
        'MCC': [mcc],
        'tpcode': [postcode],
        'NET_REVENUE': [revenue]
    })
    pred = prod_pipeline.predict(new_data)
    probs = prod_pipeline.predict_proba(new_data)[0]
    encoded_prods = np.argsort(probs)[::-1]
    
    prod_codes = [prod_encoder.inverse_transform([i]) for i in encoded_prods]
    # print(prod_codes)
    return prod_codes

@tool
def recommend_pricing_wrapper(product_code: str, mis_division: str, mcc: int, postcode: int, revenue: float) -> str:
    """
    Recomend merchant pricing based on input criteria
    Args:
        product_code: merchant product code
        mis_division: the market division. Valid values are RBS, BB, IB&M
        mcc: MCC / Merchant Category Code
        postcode: postcode of trading address
        revenue: monthly revenue / ternover in AUD

    Returns:
        The recommended pricing plan.
    """
    kwargs = locals()
    return recommend_pricing(**kwargs)

def recommend_pricing(product_code, mis_division, mcc, postcode, revenue):
    new_data = pd.DataFrame({
        'PRODUCT_CODE': [product_code],
        'MIS_DIVISION': [mis_division],
        'MCC': [mcc],
        'tpcode': [postcode],
        'NET_REVENUE': [revenue]
    })
    predictions = price_pipeline.predict(new_data)
    pricing = price_label_encoder.inverse_transform(predictions)
    return pricing
