import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv('customer_data.csv')
    X = df[['Age', 'MonthlySpend', 'SessionDuration']]
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    joblib.dump(clf, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump((X_test_scaled, y_test), 'test_data.pkl')
    print("Model trained and artifacts saved.")

if __name__ == "__main__":
    train()