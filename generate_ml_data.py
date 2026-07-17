import pandas as pd
import numpy as np

def create_dataset():
    np.random.seed(42)
    n = 400
    
    age = np.random.randint(18, 65, size=n)
    spend = np.random.uniform(10, 300, size=n)
    duration = np.random.uniform(5, 60, size=n)
    
    noise = np.random.normal(0, 1, size=n)
    val = (age * 0.02) + (spend * 0.015) + (duration * 0.04) - 3.5 + noise
    target = (val > 0).astype(int)
    
    data = {
        'Age': age,
        'MonthlySpend': spend,
        'SessionDuration': duration,
        'Outcome': target
    }
    
    df = pd.DataFrame(data)
    df.to_csv('customer_data.csv', index=False)
    print("Dataset saved to customer_data.csv")

if __name__ == "__main__":
    create_dataset()