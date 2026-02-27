import os
import numpy as np
import h5py
from PIL import Image

# init img route
IMG_ROOT = "./data/fer2013"

# processed img route
H5_OUTPUT_PATH = "./data/fer2013.h5"

EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

SPLITS = ["train", "test"]

def read_fer_image(img_path):
    """
    read the fer2013 image and convert it to a 1D array of pixel values (grayscale)
    return the 1D array of pixel values (grayscale) as a numpy array of type uint8
    shape(2304,) since the original image is 48x48 pixels, pixel values are in the range [0, 255]
    """
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img).flatten()  # Flatten to 1D array
        return img_array.astype(np.uint8)
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None


def collect_fer_data():
    """
    collect the fer2013 data and labels
    """
    data_dict = {
        "train_pixels": [], "train_labels": [],
        "test_pixels": [], "test_labels": [],
    }

    for split in SPLITS:
        split_path = os.path.join(IMG_ROOT, split)

        for emotion, labels in EMOTION_MAP.items():
            emotion_path = os.path.join(split_path, emotion)

            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                img_array = read_fer_image(img_path)

                if img_array is not None:
                    data_dict[f"{split}_pixels"].append(img_array)
                    data_dict[f"{split}_labels"].append(labels)

    # after iterating all splits, convert lists to numpy arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key], dtype=np.uint8)

    return data_dict


def save_to_h5(data_dict):
    """
    save the collected date and labels to a h5 file
    """
    with h5py.File(H5_OUTPUT_PATH, 'w') as h5_file:
        for key, value in data_dict.items():
            # pixel values are stored as uint8, labels are stored as uint64
            dtype = np.uint8 if 'pixels' in key else np.uint64
            h5_file.create_dataset(key, data=value, dtype=dtype, compression="gzip")
            print(f"Saved {key} with shape {value.shape} and dtype {dtype} to {H5_OUTPUT_PATH}")
        


if __name__ == "__main__":
    try:
        # collect the fer2013 data and labels, then save to h5 file
        fer_data = collect_fer_data()
        save_to_h5(fer_data)
    except Exception as e:
        print(f"Error: {e}")