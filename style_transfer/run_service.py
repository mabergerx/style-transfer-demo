from style_transfer.image_io import *
from style_transfer.load_model import load_model
from style_transfer.nn_definition import *


def transfer_style(content_image, style_image, size=1024, model=load_model()):
    # unsqueeze puts a dimension of 1 at the 0'th dimension of tensor:
    # effectively it puts an extra list around the existing array.
    # print("CONTENT IMAGE:", content_image)
    content_image = tensor_load_rgbimage(content_image, size=size, keep_asp=True).unsqueeze(0)
    print("Loaded content")
    # Scale seems to have no effect on the output image...
    style = tensor_load_rgbimage(style_image, size=size).unsqueeze(0)
    print("Loaded style")
    style = preprocess_batch(style)

    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    model.setTarget(style_v)
    output = model(content_image)
    return tensor_save_bgrimage(output.data[0], False)


# print(transfer_style('venice-boat.jpg', 'starrynight.jpg'))