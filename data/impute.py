import argparse

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, IterativeImputer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel',  type=str,   default="../../excel_data/filter_clinlab.xlsx")
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

def main(args):
    df = pd.read_excel(args.excel)
    dropped_rows = df[df["Surgery"].astype(str).str.contains(r'[a-zA-Z]', na=False)]
    df = df[~df["Surgery"].astype(str).str.contains(r'[a-zA-Z]', na=False)]

    df["AJCC Pathologic Stage Group"] = df["AJCC Pathologic Stage Group"].map(lambda x: int(str(x)[0]) if pd.notnull(x) else x)
    dropped = df[["Exclusion", "Vital Status", "Cancer Status"]]
    df = df.drop(columns=["Exclusion", "Vital Status", "Cancer Status"])

    imputed = impute(df)
    imputed[["Exclusion", "Vital Status", "Cancer Status"]] = dropped[["Exclusion", "Vital Status", "Cancer Status"]]
    imputed = pd.concat([imputed, dropped_rows])
    imputed = imputed.sort_values(by="Patient ID")

    imputed.to_excel(args.out, index=False)

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    main()