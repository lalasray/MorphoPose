import json
import pandas as pd
import os

BONES_JSON_PATH = os.path.join(os.path.dirname(__file__), 'bones.json')
CSV_PATH = r"coordinates_2d_Akita_Albedo_A_Pose.csv"

def get_dog_name_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    for i, part in enumerate(parts):
        if part.lower() == '2d' and i + 1 < len(parts):
            return parts[i + 1].lower()
    return None

# Load bones superset for all species
def load_bones_superset(json_path=BONES_JSON_PATH):
    with open(json_path, 'r') as f:
        bones = json.load(f)
    return bones

# Load pose data
def load_pose_bones(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={col: col.lower() for col in df.columns})
    if 'bone' not in df.columns:
        raise ValueError("CSV must contain a 'bone' column.")
    return set(df['bone'].unique())

# Generate joint presence mask for a species, using 'all' as superset
def generate_joint_presence_mask(all_bones, pose_bones):
    return [1 if bone in pose_bones else 0 for bone in all_bones]

if __name__ == "__main__":
    bones_superset = load_bones_superset()
    all_bones = bones_superset['all']
    dog_type = get_dog_name_from_filename(CSV_PATH)
    pose_bones = load_pose_bones(CSV_PATH)
    mask = generate_joint_presence_mask(all_bones, pose_bones)
    print(f"{dog_type}: {mask}")
