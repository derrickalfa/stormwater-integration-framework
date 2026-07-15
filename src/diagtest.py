import pandas as pd
v4 = pd.read_csv(r"C:\Users\Acer\Desktop\PolyU\Research\Data\ML_Aggregated_Analysis\manhole_data_FIXED_maintenance_v4_TEMPORAL_RENAMED.csv", nrows=5, low_memory=False)
print([c for c in v4.columns])