import os
import pandas as pd
from collections import Counter

def count_bone_occurrences(directory):
    bone_counter = Counter()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and '3d' in file.lower():
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'Bone' in df.columns:
                        bones = df['Bone'].dropna()
                        bone_counter.update(bones)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    print("Bone occurrence counts across all CSVs with '3d' in the name:")
    for bone, count in bone_counter.most_common():
        print(f"{bone}: {count}")

# Example usage
count_bone_occurrences(r"/home/lala/Documents/Data/MorphPose")
