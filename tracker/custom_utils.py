import os, exifread, re, glob, tqdm
import numpy as np
from PIL import Image
from functools import reduce

def format_tracks(tracks, image_path, dateTime):
    if tracks:
        return "\n".join([image_path + "\t" + dateTime + "\t" + i.__repr__() for i in tracks]) + "\n"
    else:
        return f'{image_path}\t{dateTime}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n'

class Class_handler:
    def __init__(self, classes = ["Bud", "Flower", "Withered", "Immature", "Mature"], 
                 colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"], 
                 thresholds = [0.4, 0.4, 0.4, 0.4, 0.4]) -> None:
        if len(classes) != len(colors) or len(classes) != len(thresholds):
            raise ValueError("Classes, colors and thresholds must have the same length")
        self.classes = classes
        self.colors = colors
        self.thresholds = thresholds
        self.cls_to_ind = {k : i for i, k in enumerate(self.classes)}
        self.cls_to_col = {k : v for k, v in zip(self.classes, self.colors)}
        self.cls_thresh = {k : v for k, v in zip(self.classes, self.thresholds)}
    
    def normalize_score(self, cls, score):
        # Returns the score between 0 and 1 if it is above the threshold (such that threshold -> 0 and 1 -> 1), otherwise returns None
        if cls not in self.classes:
            raise ValueError(f"Class {cls} not in {self.classes}")
        if score > self.cls_thresh[cls]:
            return 1/2 + (score - self.cls_thresh[cls]) / (2 * (1 - self.cls_thresh[cls]))
        else:
            return None
        
    def get_color(self, cls):
        if cls in self.classes:
            return self.cls_to_col[cls]
        else:
            raise ValueError(f"Class {cls} not in {self.classes}")
        
    def get_index(self, cls):
        if cls in self.classes:
            return self.cls_to_ind[cls]
        else:
            raise ValueError(f"Class {cls} not in {self.classes}")

class DUMMY_args:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)

class Raw_data(list):
    def __init__(self, dir = "../Raw_data/") -> None:
        self.dir = dir
        self.series_dict = {re.sub("_24H|_6H","",re.sub("-", "_", i)) : i for i in os.listdir(dir) if os.path.isdir(dir + i)}
        self._index = 0
        
        super().__init__(self.series_dict.keys())
        
    def get_series(self, series):
        for ind, i in enumerate(self.series_dict.keys()):
            if i == series:
                self._index = ind
                return [i]
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        next = self[self._index]
        self._index += 1
        return next
    
    def __getitem__(self, ind):
        self._index = ind
        return super().__getitem__(ind)


class ImageSeries(list):
    @classmethod
    def from_series(cls, dir, series, post = "/**", pbar = None):
        return cls(
            raw = Raw_data(dir),
            series = series,
            post = post,
            pbar = pbar
        )
    
    def __init__(self, 
                 raw, series, 
                 downscaling_factor = 4,
                 post = "/**", 
                 pbar = None,
                 subsample = 1) -> None:
        self.pbar = pbar        
        self.series = series
        self.image_paths, self.dates = self.sorted_images(glob.glob(raw.dir + raw.series_dict[series] + post))
        if subsample and subsample > 1:
            self.image_paths = self.image_paths[::subsample]
        self.d = downscaling_factor
        self._index = 0
        self._max = len(self.image_paths)
        self._iseries = raw._index
        self._nseries = len(raw)
        self.current_path = None
        
        if self.pbar is not None:
            self.pbar.reset(total=self._max)
            self.pbar.set_description(self.pbar_desc(self.series, self._iseries, self._nseries, "-"))
        
        super().__init__(self.image_paths)
    
    @staticmethod
    def sorted_images(srcs):
        dates = []
        
        for src in srcs:
            with open(src, "rb") as f:
                date = str(exifread.process_file(f)["EXIF DateTimeOriginal"])
                date = [int(i) for i in re.findall("[0-9\.-]+", date)]
                dates.append(date)
                
        return [i for _, i in sorted(zip(dates, srcs))], [":".join([str(j) for j in i]) for i in sorted(dates)]
    
    @staticmethod
    def pbar_desc(s, si, si_max, n):
        return f"Tracking objects on images (series {s} ({si}/{si_max}), {n} objects found)"
    
    def num_detected(self, n):
        self.pbar.set_description(self.pbar_desc(self.series, self._iseries, self._nseries, n))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index >= self._max:
            raise StopIteration
        next = self[self._index]
        
        returnValue = np.array(Image.open(next).reduce(self.d)), self.dates[self._index]
        
        if self.pbar is not None:
            self.pbar.update(1)
            if self._index == 0:
                self.pbar.set_description(self.pbar_desc(self.series, self._iseries, self._nseries, "-"))
            
        self._index += 1
        self.current_path = re.search("[^/]+$", next)[0]
        
        return returnValue
    
    def __len__(self):
        return self._max
    