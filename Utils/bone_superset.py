import os
import pandas as pd

def find_unique_bones(directory):
    unique_bones = set()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and '3d' in file.lower():
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'Bone' in df.columns:
                        bones = df['Bone'].dropna().unique()
                        unique_bones.update(bones)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    print("Unique 'Bone' entries in CSV files with '3d' in their names:")
    for bone in sorted(unique_bones):
        print(bone)

# Example usage
find_unique_bones(r"/home/lala/Documents/Data/MorphPose/output_shepherd")
