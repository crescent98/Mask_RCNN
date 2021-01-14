import python_pachyderm
from PIL import Image
from tqdm import tqdm
from utils import *
import sys

class PachySemanticDataset():
    def __init__(self, commit, path_prefix, pachy_host="local_host", pachy_port="30650",
               local_root='/data'):
        self.commit = commit
        self.path_prefix = path_prefix
        self.client = python_pachyderm.Client(host=pachy_host, port=pachy_port)
        self.image_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                               for res in self.client.glob_file(commit, path_prefix + "images/*")]
        self.label_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                               for res in self.client.glob_file(commit, path_prefix + "labels/*")]
        self.local_root = local_root
        self.download_data_from_pachyderm(self.image_path_lst, self.path_prefix + "images/*")
        self.download_data_from_pachyderm(self.label_path_lst, self.path_prefix + "labels/*")
        self.dataset_root = self.local_root + '/data/'
    def download_data_from_pachyderm(self, path_lst, glob):
        print("Downloading data into worker")
        idx = 0
        continued = False
        current_size = 0
        for chunk in self.client.get_file(self.commit, glob):
          local_path = join_pachy_path(self.local_root, path_lst[idx]['path'])
          if not continued:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
          with open(local_path, "ab" if continued else "wb") as local_file:
            local_file.write(chunk)
            current_size += len(chunk)
          if current_size == path_lst[idx]["size"]:
            idx += 1
            continued = False
            current_size = 0
          elif current_size < path_lst[idx]["size"]:
            continued = True
          else:
            raise IOError("Wrong chunk size")
        print(f"Downloaded {idx} files")
    


