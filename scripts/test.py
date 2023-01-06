# from PIL import Image
import csv
import numpy as np

# imgpath = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/CITYSCAPES_DATASET/gtFine/train/hamburg/hamburg_000000_044251_gtFine_color.png'

# img = Image.open(imgpath)
# print(type(img))

# img.show()
test_arr = np.array([5, 4, 3, 2, 1])

filepath = 'C:/Users/noahv/OneDrive/MyProjects2022Onward/Ongoing/GithubPublicRepositories/UNet-Multiclass/test.csv'


np.savetxt(filepath, test_arr)
# with open(filepath, mode='w') as csvwriter:
# logwrite = csv.writer(csvwriter, delimiter=',')
# logwrite.writerows(test_arr)
