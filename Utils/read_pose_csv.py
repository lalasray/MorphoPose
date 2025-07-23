import os
import pandas as pd
import json

# Path to the 2D pose CSV file
CSV_PATH = r"/home/lala/Documents/Data/MorphPose/output_akita/coordinates_2d_Akita_Albedo_A_Pose.csv"
BONES_JSON_PATH = os.path.join(os.path.dirname(__file__), 'bones.json')

def get_dog_name_and_action_from_filename(filename):
    """
    Extracts the dog name and action from the filename.
    Example: 'coordinates_2d_Akita_Albedo_A_Pose.csv' -> ('Akita', 'A_Pose')
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    dog_name = None
    action = None
    for i, part in enumerate(parts):
        if part.lower() == '2d' and i + 1 < len(parts):
            dog_name = parts[i + 1]
            # Action is everything after the dog name, joined by underscores, minus extension
            if i + 2 < len(parts):
                action = '_'.join(parts[i + 2:]).rsplit('.', 1)[0]
            break
    return dog_name, action

def read_pose_csv(csv_path=CSV_PATH):
    """
    Reads a pose CSV file and returns a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

def load_bones_superset(json_path=BONES_JSON_PATH):
    """
    Loads the superset of bones from a JSON file.
    """
    with open(json_path, 'r') as f:
        bones = json.load(f)
    return bones

def load_pose_bones(df):
    """
    Loads the set of bones present in the pose DataFrame.
    """
    df = df.rename(columns={col: col.lower() for col in df.columns})
    if 'bone' not in df.columns:
        raise ValueError("CSV must contain a 'bone' column.")
    return set(df['bone'].unique())

def generate_joint_presence_mask(all_bones, pose_bones):
    """
    Generates a joint presence mask indicating which bones are present in the pose.
    """
    return [1 if bone in pose_bones else 0 for bone in all_bones]

def create_datapoints_by_camera_and_frame(df, csv_path=CSV_PATH, mask=None):
    """
    Creates a datapoint (dict) for each camera, frame, and focal length in the DataFrame.
    Supports both 2D (x, y) and 3D (x, y, z) pose files based on filename and columns.
    """
    df = df.rename(columns={col: col.lower() for col in df.columns})
    dog_name, dog_action = get_dog_name_and_action_from_filename(csv_path)
    file_name = os.path.basename(csv_path)
    is_3d = '3d' in file_name.lower()
    required_cols_2d = {'frame', 'camera', 'bone', 'x', 'y', 'focal length'}
    required_cols_3d = {'frame', 'camera', 'bone', 'x', 'y', 'z', 'focal length'}
    if is_3d and required_cols_3d.issubset(df.columns):
        datapoints = []
        for (frame, camera, focal_length), group in df.groupby(['frame', 'camera', 'focal length']):
            pose = group[['bone', 'x', 'y', 'z']].to_dict(orient='records')
            datapoints.append({
                'dog_name': dog_name,
                'file_name': file_name,
                'dog_action': dog_action,
                'camera': camera,
                'frame': frame,
                'focal_length': focal_length,
                'pose': pose,
                'joint_presence_mask': mask
            })
        return datapoints
    elif not is_3d and required_cols_2d.issubset(df.columns):
        datapoints = []
        for (frame, camera, focal_length), group in df.groupby(['frame', 'camera', 'focal length']):
            pose = group[['bone', 'x', 'y']].to_dict(orient='records')
            datapoints.append({
                'dog_name': dog_name,
                'file_name': file_name,
                'dog_action': dog_action,
                'camera': camera,
                'frame': frame,
                'focal_length': focal_length,
                'pose': pose,
                'joint_presence_mask': mask
            })
        return datapoints
    else:
        print("CSV must contain the required columns for 2D or 3D pose (case-insensitive).")
        return []

def create_datapoints_with_2d_3d(df2d, df3d, csv_path_2d, csv_path_3d, mask=None):
    """
    For each unique (frame, camera, focal length) in 2D, create a datapoint with:
      - 2D pose for that (frame, camera, focal length)
      - 3D pose for that frame (not camera/focal length dependent, only one entry per bone)
    Fills missing camera and focal length in 3D data from 2D data if needed.
    """
    df2d = df2d.rename(columns={col: col.lower() for col in df2d.columns})
    df3d = df3d.rename(columns={col: col.lower() for col in df3d.columns})
    dog_name, dog_action = get_dog_name_and_action_from_filename(csv_path_2d)
    file_name_2d = os.path.basename(csv_path_2d)
    file_name_3d = os.path.basename(csv_path_3d)
    required_cols_2d = {'frame', 'camera', 'bone', 'x', 'y', 'focal length'}
    required_cols_3d = {'frame', 'bone', 'x', 'y', 'z'}
    # If camera/focal length missing in 3D, fill from 2D by merging on frame+bone
    if not {'camera', 'focal length'}.issubset(df3d.columns):
        df3d = pd.merge(
            df3d,
            df2d[['frame', 'bone', 'camera', 'focal length']].drop_duplicates(),
            on=['frame', 'bone'],
            how='left'
        )
    # Remove duplicate (frame, bone) in 3D, keep first occurrence
    df3d = df3d.drop_duplicates(subset=['frame', 'bone'])
    if required_cols_2d.issubset(df2d.columns) and required_cols_3d.issubset(df3d.columns):
        datapoints = []
        # Precompute 3D pose for each frame (one entry per bone)
        pose3d_by_frame = {
            frame: group[['bone', 'x', 'y', 'z']].to_dict(orient='records')
            for frame, group in df3d.groupby('frame')
        }
        for (frame, camera, focal_length), group2d in df2d.groupby(['frame', 'camera', 'focal length']):
            pose_2d = group2d[['bone', 'x', 'y']].to_dict(orient='records')
            pose_3d = pose3d_by_frame.get(frame, [])
            datapoints.append({
                'dog_name': dog_name,
                'file_name_2d': file_name_2d,
                'file_name_3d': file_name_3d,
                'dog_action': dog_action,
                'camera': camera,
                'frame': frame,
                'focal_length': focal_length,
                'pose_2d': pose_2d,
                'pose_3d': pose_3d,
                'joint_presence_mask': mask
            })
        return datapoints
    else:
        print("Both 2D and 3D CSVs must contain the required columns (case-insensitive). Columns in 2D:", df2d.columns, ", Columns in 3D:", df3d.columns)
        return []

if __name__ == "__main__":
    CSV_PATH_2D = r"/home/lala/Documents/Data/MorphPose/output_akita/coordinates_2d_Akita_Albedo_A_Pose.csv"
    CSV_PATH_3D = r"/home/lala/Documents/Data/MorphPose/output_akita/coordinates_3d_Akita_Albedo_A_Pose.csv"
    df2d = read_pose_csv(CSV_PATH_2D)
    df3d = read_pose_csv(CSV_PATH_3D)
    bones_superset = load_bones_superset()
    all_bones = bones_superset['all']
    pose_bones = load_pose_bones(df2d)
    mask = generate_joint_presence_mask(all_bones, pose_bones)
    datapoints = create_datapoints_with_2d_3d(df2d, df3d, CSV_PATH_2D, CSV_PATH_3D, mask)
    print(f"Total datapoints: {len(datapoints)}")
    print(datapoints[:1])
