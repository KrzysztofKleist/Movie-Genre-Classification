import os

parent_folder = "raw_frames/frames"

parent_count = 0

# Loop over subfolders in the parent folder
for subfolder in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Loop over sub-subfolders in the subfolder
        count = 0
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            # if os.path.isdir(subsubfolder_path):
            #     count += 1

            for frame in os.listdir(subsubfolder_path):
                frame_path = os.path.join(subsubfolder_path, frame)
                if os.path.exists(frame_path):
                    count += 1

    parent_count += count

print(f"Number of links processed: {parent_count}")
