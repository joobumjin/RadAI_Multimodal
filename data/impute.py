import argparse

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, IterativeImputer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel',  type=str,   default="../../excel_data/Combined_Clinical_Lab.xlsx")
    parser.add_argument('--out',    type=str,   default="../../excel_data/Imputed_ClinLab.xlsx")

    return parser

def impute(df):
    # 1. Initialize Scaler and Imputer
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=3)

    # 2. Scale the data
    scaled_data = scaler.fit_transform(df)

    # 3. Impute the scaled data
    imputed_data = imputer.fit_transform(scaled_data)

    # 4. Inverse transform to get original scales back
    final_data = scaler.inverse_transform(imputed_data)

    # 5. Convert back to DataFrame
    return pd.DataFrame(final_data, columns=df.columns)

def main():
    parser  = get_args_parser()
    args    = parser.parse_args()

    df = pd.read_excel(args.excel)
    imputed = impute(df)
    imputed.to_excel(args.out, index=False)


if __name__ == '__main__':
    main()