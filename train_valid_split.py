import shutil
import os
import numpy as np
import argparse

def get_files_from_folder(path):

    files = os.listdir(path)
    list_files = np.asarray(files)
    np.random.shuffle(list_files)
    return list_files

def main(path_to_data, path_to_validate_data, train_ratio):
    files = get_files_from_folder(path_to_data)
    data_counter = len(files)
    validation_counter = np.round(data_counter * (1 - train_ratio))

    path_to_save = path_to_validate_data

    #creates dir
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # moves data
    for j in range(int(validation_counter)):
        dst = os.path.join(path_to_save, files[j])
        src = os.path.join(path_to_data, files[j])
        shutil.move(src, dst)


def parse_args():
  parser = argparse.ArgumentParser(description="Dataset divider")
  parser.add_argument("--data_path", required=True,
    help="Path to data")
  parser.add_argument("--test_data_path_to_save", required=True,
    help="Path to test data where to save")
  parser.add_argument("--train_ratio", required=True,
    help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.data_path, args.test_data_path_to_save, float(args.train_ratio))