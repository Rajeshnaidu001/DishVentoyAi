import sys
import os
import json
import pandas as pd

# Append backend to path to import app module
sys.path.append('/Users/rajeshchinta/dishventoryAi/backend')
import app

def main():
    print("Reading test.csv...")
    df = pd.read_csv('/Users/rajeshchinta/dishventoryAi/test.csv')
    
    print("Computing Pepperoni Pizza sales forecast via Prophet...")
    result = app.forecast_pepperoni(df)
    
    print("Writing demo JSON to frontend/demo_forecast.json...")
    out_path = '/Users/rajeshchinta/dishventoryAi/frontend/demo_forecast.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"Success! Generated demo JSON at {out_path}")

if __name__ == '__main__':
    main()
