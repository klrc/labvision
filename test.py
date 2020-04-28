import labvision.transforms as transforms
import labvision


root = '/home/sh/Desktop/Research/external/Flickr_LDL'
emod = labvision.datasets.FlickrLDL(root=root)
print(emod.__getitem__(5))
