import cv2 

##############################################################

# These parameter values are indicative. You should choose your own 
# according to properties of the method you want to demonstrate

h = 35
templateWindowSize = 7 
searchWindowSize = 35

##############################################################

img = cv2.imread('dice-extremeNoise.png')

dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)

cv2.imwrite('denoised-dice-extremeNoise.png', dst)
