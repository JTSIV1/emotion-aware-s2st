import os
import requests
import tarfile
import shutil
from moviepy import VideoFileClip

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
    if os.path.exists(dest):
        print(f"Archive {dest} found. Proceeding to extraction.")
        return
    
    print("downloading raw data...")
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("download complete")

def process_and_cleanup(extract_path, final_data_path):
    # Based on your screenshot, the sub-archives are inside 'MELD.Raw'
    meld_raw_dir = os.path.join(extract_path, "MELD.Raw")
    splits = ['train', 'dev', 'test']
    
    # 1. Preserve the CSV labels for your Emotion-Aware S2ST research
    print("Moving CSV label files to ./data...")
    for csv_file in [f for f in os.listdir(meld_raw_dir) if f.endswith('.csv')]:
        shutil.move(os.path.join(meld_raw_dir, csv_file), os.path.join(final_data_path, csv_file))

    for split in splits:
        inner_tar_name = f"{split}.tar.gz"
        inner_tar_path = os.path.join(meld_raw_dir, inner_tar_name)
        
        if not os.path.exists(inner_tar_path):
            print(f"Skipping {split}: Archive {inner_tar_name} not found.")
            continue

        # 2. Extract the sub-archive (e.g., train.tar.gz)
        print(f"Extracting {inner_tar_name}...")
        with tarfile.open(inner_tar_path, "r:gz") as tar:
            # This usually extracts into a folder simply named 'train', 'dev', or 'test'
            tar.extractall(path=meld_raw_dir)

        # Path where the .mp4 files just landed
        temp_video_folder = os.path.join(meld_raw_dir, split)
        output_folder = os.path.join(final_data_path, split)

        # 3. Batch process and delete immediately
        print(f"Converting {split} clips to 16kHz audio...")
        if os.path.exists(temp_video_folder):
            for filename in os.listdir(temp_video_folder):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(temp_video_folder, filename)
                    audio_path = os.path.join(output_folder, filename.replace(".mp4", ".wav"))
                    
                    try:
                        video = VideoFileClip(video_path)
                        # Ensuring 16kHz mono for Wav2Vec compatibility
                        video.audio.write_audiofile(
                            audio_path, 
                            fps=16000, 
                            nbytes=2, 
                            codec='pcm_s16le', 
                            verbose=False, 
                            logger=None
                        )
                        video.close()
                        
                        # DELETE the video file to free up space
                        os.remove(video_path)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
            
            # Clean up the now-empty temporary folder
            shutil.rmtree(temp_video_folder)
        
        # 4. DELETE the sub-archive after processing
        os.remove(inner_tar_path)
        print(f"Finished processing and cleaning up {split} set.")


if __name__ == "__main__":
    # setup_directories(DATA_DIR)

    # download_file(MELD_URL, TEMP_TAR)
    
    # print("extracting data...")
    # with tarfile.open(TEMP_TAR, "r:gz") as tar:
    #     tar.extractall(path=RAW_EXTRACT_DIR)
    # os.remove(TEMP_TAR)

    process_and_cleanup(RAW_EXTRACT_DIR, DATA_DIR)

    # print("cleanup...")
    # shutil.rmtree(RAW_EXTRACT_DIR)

    # print(f"done! data in {os.path.abspath(DATA_DIR)}")
