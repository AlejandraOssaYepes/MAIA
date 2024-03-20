import cv2 
import matplotlib.pyplot as plt
import numpy as np 

class Image:

    def __init__(self, path_image, new_width, new_lenght, smooth_kernel) -> None:
        self.path_image = path_image
        self.image = None
        self.new_width = new_width
        self.new_lenght = new_lenght
        self.smooth_kernel =  smooth_kernel
        self.Preprocessed_image = None 


    def Read_Image(self):
        self.image = cv2.imread(self.path_image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def Image_resize_Smooth(self):
        n, m, _ = self.image.shape
        reduce_image = cv2.resize(self.image, (self.new_width, self.new_width), interpolation= cv2.INTER_AREA)
        reduce_image = reduce_image

        imagen_suavizada = cv2.GaussianBlur(reduce_image, self.smooth_kernel, 0)  
        
        smooth_image  = imagen_suavizada/255.0

        self.Preprocessed_image = smooth_image 


    def Pipeline(self):
        self.Read_Image()
        self.Image_resize_Smooth()

           
    def Show_Image(self, type_image):

        if type_image == 'Original':
            plt.imshow(self.image)
            plt.show()
        elif  type_image == 'Preprocessed':
            plt.imshow(self.Preprocessed_image)
            plt.show()
        else: 
            raise('Error in the type of image')



     


     



