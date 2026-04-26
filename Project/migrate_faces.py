import os
import shutil

FACES_DIR = "faces"

def migrate():
    if not os.path.exists(FACES_DIR):
        return
    
    for filename in os.listdir(FACES_DIR):
        if os.path.isfile(os.path.join(FACES_DIR, filename)) and (filename.endswith(".jpg") or filename.endswith(".png")):
            name = filename.split('_')[0]
            user_dir = os.path.join(FACES_DIR, name)
            
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            # Use original as the first sample
            new_path = os.path.join(user_dir, "1.jpg")
            shutil.move(os.path.join(FACES_DIR, filename), new_path)
            print(f"Migrated {filename} -> {new_path}")

if __name__ == "__main__":
    migrate()
