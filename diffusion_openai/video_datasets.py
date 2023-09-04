from random import sample
from PIL import Image, ImageSequence
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import av

def _parse_filename(filename):
    """
    Parse the filename into its components based on index.
    """
    timestep = filename[:3]
    row = filename[3]
    column = filename[4:6]
    field = filename[6:8]
    channel = filename[8:10]
    return timestep, row, column, field, channel

def _list_tif_files_recursively(data_dir):
    """
    List all .TIF files in the directory recursively.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["tif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_tif_files_recursively(full_path))
    return results

class VideoDataset_tif(Dataset):
    def __init__(
            self, resolution, video_paths, classes=None, shard=0, 
            num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None
        if classes is not None:
            self.local_classes = classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        filename = bf.basename(path)
        timestep, row, column, field, channel = _parse_filename(filename)
        arr_list = []
        video_container = av.open(path)
        n = video_container.streams.video[0].frames
        frames = [i for i in range(n)]
        if n > self.seq_len:
            start = np.random.randint(0, n-self.seq_len)
            frames = frames[start:start + self.seq_len]
        for id, frame_av in enumerate(video_container.decode(video=0)):
            if (id not in frames):
                continue
            frame = frame_av.to_image()
            while min(*frame.size) >= 2 * self.resolution:
                frame = frame.resize(
                    tuple(x // 2 for x in frame.size), resample=Image.BOX
                )
            scale = self.resolution / min(*frame.size)
            frame = frame.resize(
                tuple(round(x * scale) for x in frame.size),
                resample=Image.BICUBIC
            )

            if self.rgb:
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.array(frame.convert("L"))
                arr = np.expand_dims(arr, axis=2)
            crop_y = (arr.shape[0] - self.resolution) // 2
            crop_x = (arr.shape[1] - self.resolution) // 2
            arr = arr[crop_y : crop_y + self.resolution, 
                      crop_x : crop_x + self.resolution]
            arr = arr.astype(np.float32) / 127.5 - 1
            arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        # fill in missing frames with 0s
        if arr_seq.shape[1] < self.seq_len:
            required_dim = self.seq_len - arr_seq.shape[1]
            fill = np.zeros((3, required_dim, self.resolution, 
                             self.resolution))
            arr_seq = np.concatenate((arr_seq, fill), axis=1)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        out_dict["timestep"] = timestep
        out_dict["row"] = row
        out_dict["column"] = column
        out_dict["field"] = field
        out_dict["channel"] = channel
        return arr_seq, out_dict

def preprocess_tif(arr):
    """
    Preprocess the 16-bit TIF image to be suitable for the diffusion model.
    """
    # Normalize to [0, 1]
    arr = arr / 65535.0
    # Convert to float32
    arr = arr.astype(np.float32)
    return arr

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, 
    deterministic=False, rgb=True, seq_len=20
):
    """
    For a dataset, create a generator over (videos, kwargs) pairs.

    Each video is an NCLHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which frames are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_tif_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Use the class labels extracted from the filename using _parse_filename
        class_labels = [_parse_filename(bf.basename(path))[3] 
                        for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_labels)))}
        classes = [sorted_classes[x] for x in class_labels]
    
    entry = all_files[0].split(".")[-1]
    if entry in ["tif"]:
        dataset = VideoDataset_tif(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            rgb=rgb,
            seq_len=seq_len
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=16, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=16, drop_last=True
        )
    while True:
        yield from loader