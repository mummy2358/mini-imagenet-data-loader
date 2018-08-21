# the data loader for imagenet
# by mym
# images are resized so that shorter side is 256pix, then cropped according to a given size
import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from os import listdir as ldir

class imagenet_loader:
  def __init__(self,class_seq=None):
    # create a data generator only for given classes
    # class_seq is a list of string like [n014xxxxx,n013xxxxx,...]
    self.class_ids=self.get_class_ids()
    if class_seq!=None:
      self.class_ids=class_seq
    self.image_names={}
    self.get_image_names()
    self.img_coordinates=[]
    self.get_image_coordinates()
    self.sample_counter=0
    self.batch_counter=0
    self.class_num=len(self.class_ids)
  
  def get_class_ids(self):
    # get class ids(folder names)
    # thanks to https ://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    class_id_original=open("./map_clsloc.txt","r").read().split("\n")
    class_id=[]
    for line in class_id_original:
      class_id.append(line[:9])
    return class_id
  
  def get_image_names(self):
    # get the listed image names for each folder in training set
    for class_name in self.class_ids:
      self.image_names[class_name]=ldir("./train/"+class_name)
  
  def get_image_coordinates(self):
    # get all possible 2d image ids: [class_id,image_id]---[str,int]
    # for later random shuffle
    for class_name in self.image_names:
      for i in range(len(self.image_names[class_name])):
        self.img_coordinates.append([class_name,i])
  
  def preprocess(self,image,shorter=256):
    # resize given image so that shorter edge is the given size
    # image shape: [H,W,C]
    sh=np.shape(image)
    if sh[0]<=sh[1]:
      img=resize(image,output_shape=(shorter,int(sh[1]*shorter/sh[0])))
    else:
      img=resize(image,output_shape=(int(sh[0]*shorter/sh[1]),shorter))
    return img
  
  def random_crop(self,image,size=224):
    # crop a random square area of the given image
    # image shape: [H,W,C]
    sh=np.shape(image)
    shifted_h=np.random.randint(low=0,high=sh[0]-size)
    shifted_w=np.random.randint(low=0,high=sh[1]-size)
    return image[shifted_h:(shifted_h+size),shifted_w:(shifted_w+size),:]
  
  def train_next_batch(self,batch_size,batch_per_epoch=float("inf")):
    batch_x=[]
    batch_y=[]
    np.random.shuffle(self.img_coordinates)
    while True:
      class_name=self.img_coordinates[self.sample_counter][0]
      image_name=self.image_names[self.img_coordinates[self.sample_counter][0]][self.img_coordinates[self.sample_counter][1]]
      img=io.imread("./train/"+class_name+"/"+image_name)
      img=self.preprocess(img)
      crop=self.random_crop(img)
      io.imsave("./test.png",crop)
      batch_x.append(crop)
      batch_y.append(np.eye(self.class_num)[self.class_ids.index(class_name)])
      self.sample_counter+=1
      if self.sample_counter%batch_size==0 or self.sample_counter==len(self.img_coordinates):
        yield [batch_x,batch_y]
        batch_x=[]
        batch_y=[]
      if self.batch_counter==batch_per_epoch or self.sample_counter==len(self.img_coordinates):
        np.random.shuffle(self.img_coordinates)
        self.sample_counter=0

