from os import listdir
import imageio
from pandas import read_csv
import matplotlib.pyplot as plt

DATA_DIR = '../data/'

TRAIN_IMAGE_DIR = '../data/train/images/'
TRAIN_MASK_DIR = '../data/train/masks/'

TEST_IMAGE_DIR = '../data/test/images/'


def load_train_csv():
	'''Loads the run-length encodings of the masks of the 
	training data as a pandas data frame.
	'''
	train_file = '../data/train.csv'
	return read_csv( train_file )


def load_depths_csv():
	'''Loads the depth information of each seismic image as a data frame.
	'''
	depths_file =  '../data/depths.csv'
	return read_csv( depths_file )


def list_images( folder ):
	'''List the images in a folder.
	Returns a list of image names (without file extension).
	'''
	filelist = listdir( folder )
	return [ filename[:-4] for filename in filelist ]
	

def load_img( filename ):
	'''Loads the image from the `filename` as a numpy array.
	'''
	img = imageio.imread( filename )
	return img

	
def load_train_img( img_id ):
	'''Returns the matrix representation of the train image given the id.
	'''
	filename = TRAIN_IMAGE_DIR + img_id + '.png'
	image = load_img( filename )[:,:,0]
	return image
	
	
def load_test_img( img_id ):
	'''Returns the matrix representation of the test image given the id.
	'''
	filename = TEST_IMAGE_DIR + img_id + '.png'
	image = load_img( filename )[:,:,0]
	return image
	
	
def load_train_mask( img_id ):
	'''Returns the matrix representation of the training mask image given the id.
	'''
	filename = TRAIN_MASK_DIR + img_id + '.png'
	return load_img( filename )
	

def plot( img ):
	'''Plots a grayscaled image of the 2d array `img`.
	'''
	plt.imshow( img , cmap='gray')
	plt.show()
	
	
def plot_overlay( img, overlay ):
	'''Plots a grayscaled image of the 2d array `img` and
	overlays this with a partly transparent image.
	'''
	plt.imshow( img, cmap='gray')
	plt.imshow( overlay, cmap='jet', alpha=0.5)
	plt.show()
	

