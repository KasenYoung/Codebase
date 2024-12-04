import numpy as np
import torch
import torchvision.transforms as TT
import transformers
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Optional, Tuple, Union
import os 
from tqdm import tqdm
import pandas as pd


class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = "subset-video",
        instance_caption_root:Optional[str] = "subset-video/captions/OpenVid-1M.csv",
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        

        self.instance_video_paths=os.listdir(self.instance_data_root)[:10]
        self.csv=instance_caption_root
        self.captions=self._preprocess_caption()
        print(self.captions)

        '''
        Only for test code. For formal training, we should rewrite the dataset.
        '''

        self.instance_videos=self._preprocess_data()
        
    
    def _preprocess_caption(self):
        df = pd.read_csv(self.csv)
        captions=[]
        for v in self.instance_video_paths:
            caption = df[df['video'] == v]['caption'].values
            caption=str(caption)
            captions.append(caption)
       
        return captions

    
    def __len__(self):
        return len(self.instance_video_paths)
    
    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.captions[index],
            "instance_video": self.instance_videos[index],
        }

    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )
        
        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr
    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The decord package is required for loading the video dataset. Install with pip install decord"
            )

        decord.bridge.set_bridge("torch")

        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)),
            desc="Loading progress resize and crop videos",
        )
        videos = []

        for filename in self.instance_video_paths:
            uri=os.path.join(self.instance_data_root,filename)
            video_reader = decord.VideoReader(uri)
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            frames = (frames - 127.5) / 127.5
            frames = frames.permute(0, 3, 1, 2)  # [F, C, H, W]
            progress_dataset_bar.set_description(
                f"Loading progress Resizing video from {frames.shape[2]}x{frames.shape[3]} to {self.height}x{self.width}"
            )
            frames = self._resize_for_rectangle_crop(frames)
            videos.append(frames.contiguous())  # [F, C, H, W]
            progress_dataset_bar.update(1)

        progress_dataset_bar.close()
        return videos

dataset=VideoDataset()