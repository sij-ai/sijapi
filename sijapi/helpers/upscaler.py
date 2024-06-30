from aura_sr import AuraSR
from PIL import Image

aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR")

def load_image_from_path(file_path):
    # Open image file
    return Image.open(file_path)

def upscale_and_save(original_path):
    # load the image from the path
    original_image = load_image_from_path(original_path)
    
    # upscale the image using the pretrained model
    upscaled_image = aura_sr.upscale_4x(original_image)
    
    # save the upscaled image back to the original path
    # Overwrite the original image
    upscaled_image.save(original_path)

# Now to use the function, provide the image path
upscale_and_save("/Users/sij/workshop/sijapi/sijapi/testbed/API__00482_ 2.png")
