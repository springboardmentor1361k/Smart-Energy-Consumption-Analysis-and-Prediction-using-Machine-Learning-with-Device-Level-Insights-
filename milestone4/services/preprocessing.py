from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()
    df["appliance_encoded"] = le.fit_transform(df["appliance"])
    return df, le
