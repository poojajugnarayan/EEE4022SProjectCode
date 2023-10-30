from PIL import Image

for i in range(60, 290):
    temp="Test Images/segmented/segmented_"+str(i)+".png"
    img = Image.open(temp)  # Replace with the path to your image file

    # Convert the image to 24-bit depth
    img = img.convert("RGB")

    # Save the converted image
    temp="Test Images/segmented/segmented_conv_"+str(i)+".png"
    img.save(temp)  # Replace with the desired path for the converted image
    # end for
# Open the RGB8 image
