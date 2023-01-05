from PIL import Image

imgpath = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/CITYSCAPES_DATASET/gtFine/train/hamburg/hamburg_000000_044251_gtFine_color.png'

img = Image.open(imgpath)
print(type(img))

img.show()
