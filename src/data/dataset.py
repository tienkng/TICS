import glob

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (ApplyTransformToKey, Normalize, \
                                    RandomShortSideScale, UniformTemporalSubsample)
    
MEAN  = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)
    
    
class TicDataset:
    def __init__(
        self, 
        data_path,
        img_size=224,
        num_frames=32,
        sampling_rate=2,
        start_sec = 0,
        frames_per_second = 30,
        seq_len = 5,
        split=False, 
        training_mode=False):
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.start_sec = start_sec
        self.frames_per_second = frames_per_second
        self.label_name = {'01-eye1':0, '02-eye2':1, '08-mouth2':2, '12-hand3':3, '13-face1':4, '14-face2':5, '15-face3':6}

        self._init_augs(training=training_mode)
        self.label, self.training_data, self.testing_data = self.load_dataset(data_path, split=split)
        
        
    def load_dataset(self, data_path, split=False):
        data, label = [], []
    
        for x in glob.glob(f'{data_path}/*/*'):
            # add label
            if x.endswith('xlsx'):
                label.append(x)
            # add data
            else:
                data.append(x)
        
        if split:
            train, test = train_test_split(data, test_size=0.1, shuffle=True, random_state=42)
            return label, train, test
                
        return label, data, None
    
    def _read_video_sample(self, sample):
        end_sec = self.start_sec + (self.num_frames * self.sampling_rate)/self.frames_per_second
        # Initialize an EncodedVideo helper class    
        video = EncodedVideo.from_path(file_path=sample)

        # Load the desired clip
        video_data = video.get_clip(start_sec=self.start_sec, end_sec=end_sec)
        
        return video_data

    def _init_augs(self, training):
        if training:
            self.transform = Compose(
                [
                ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.seq_len),
                        Lambda(lambda x: x / 255.0),
                        Normalize(MEAN, STD),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(self.img_size),
                        RandomHorizontalFlip(p=0.5),
                    ]
                    ),
                ),
                ]
            )
        else:
            self.transform = Compose(
                    [
                    ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize(MEAN, STD),
                            RandomCrop(self.img_size),
                        ]
                        ),
                    ),
                    ]
                )
    
    def __getitem__(self, idx:int):
        sample = self.training_data[idx]
        label = sample.split('/')[-1].split('_')[0].strip()
        video_data = self._read_video_sample(sample)
        video_data = self.transform(video_data)

        return video_data['video'], torch.tensor(self.label_name[label])
    
    def __len__(self):
        return len(self.training_data)
    
if __name__ == '__main__':
    import torch
    
    dataset = TicDataset('datahub', training_mode=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True)
    for batch, (inputs, target) in enumerate(data_loader):
        print(inputs.size(), target)
        