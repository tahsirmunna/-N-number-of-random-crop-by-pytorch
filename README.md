# N-number-of-random-crop-by-pytorch

This is a very simple code for doing random crop an image but you can choose the number of crops you want for an image. PyTorch has only 5 or 10 crops you can do. In my code, you can crop as much as you want in a random manner.



# CODE

from torchvision import transforms

def NRandomCrop(img,size, N):
  '''
  @img : image for cropping
  @size: cropping size
  @N: number of cropping
  @return: list of N number of cropped images

  '''
  tf = transforms.Compose(
        [transforms.RandomCrop(size), transforms.ToTensor()])
        
  lst_img=[] #list of cropped images as tensor
  
  for _ in range(N):
    tf_img= tf(img)
    lst_img.append(tf_img)

  return lst_img
  
  
  # Example
  #N number of random crops
    train_set = transforms.Compose(
        [(lambda crops: NRandomCrop(crops, 256, 15)) # NRandomCrop(N,M,K), Here N= images, M=crop size, K= number of crops
