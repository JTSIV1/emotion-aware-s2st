import os
import requests
import tarfile
import shutil
import subprocess

MELD_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
TEMP_TAR = "MELD_Raw.tar.gz"
RAW_EXTRACT_DIR = "./temp_meld_extract"
DATA_DIR = "./data"

def setup_directories(base_path):
    splits = ['train', 'dev', 'test']
    for split in splits:
        path = os.path.join(base_path, split)
        if not os.path.exists(path):
            os.makedirs(path)

def download_file(url, dest):    
    print("downloading raw data...")
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("download complete")

def process_and_cleanup(extract_path, final_data_path):
    meld_raw_dir = os.path.join(extract_path, "MELD.Raw")
    splits = ['train', 'dev', 'test']
    exported_name = {'train' : "train_splits", 'dev' : "dev_splits_complete", 'test' : "output_repeated_splits_test"}

    for split in splits:
        inner_tar_name = f"{split}.tar.gz"
        inner_tar_path = os.path.join(meld_raw_dir, inner_tar_name)
        
        if not os.path.exists(inner_tar_path):
            print(f"{inner_tar_name} not found.")
            continue

        print(f"extracting {inner_tar_name}...")
        with tarfile.open(inner_tar_path, "r:gz") as tar:
            tar.extractall(path=meld_raw_dir)

        temp_video_folder = os.path.join(meld_raw_dir, exported_name[split])
        output_folder = os.path.join(final_data_path, split)

        print(f"converting {split} clips to 16kHz audio...")
        if os.path.exists(temp_video_folder):
            for filename in os.listdir(temp_video_folder):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(temp_video_folder, filename)
                    audio_path = os.path.join(output_folder, filename.replace(".mp4", ".wav"))
                    
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-i', video_path,
                            '-vn', 
                            '-acodec', 'pcm_s16le',
                            '-ar', '16000', 
                            '-ac', '1', 
                            '-loglevel', 'error', 
                            audio_path
                        ], check=True)
                        
                        os.remove(video_path)
                    except Exception as e:
                        print(f"error processing {filename}: {e}")
            
            shutil.rmtree(temp_video_folder)
        
        os.remove(inner_tar_path)
        print(f"finished processing and cleaning up {split} set.")

    print("Moving CSV label files to ./data...")
    for csv_file in [f for f in os.listdir(meld_raw_dir) if f.endswith('.csv')]:
        shutil.move(os.path.join(meld_raw_dir, csv_file), os.path.join(final_data_path, csv_file))


if __name__ == "__main__":
    setup_directories(DATA_DIR)

    download_file(MELD_URL, TEMP_TAR)
    
    print("extracting data...")
    with tarfile.open(TEMP_TAR, "r:gz") as tar:
        tar.extractall(path=RAW_EXTRACT_DIR)
    os.remove(TEMP_TAR)

    process_and_cleanup(RAW_EXTRACT_DIR, DATA_DIR)

    print("cleanup...")
    shutil.rmtree(RAW_EXTRACT_DIR)

    print(f"done! data in {os.path.abspath(DATA_DIR)}")
