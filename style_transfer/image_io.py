import os
from PIL import Image
import numpy as np
import torch
import base64
import io


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    # img = Image.open(filename).convert('RGB')
    # print('Img:', img)
    # print(filename[0])
    imgdata = filename.encode('utf8').split(b';base64,')[1]

    image_ = base64.decodebytes(imgdata)

    img = Image.open(io.BytesIO(image_)).convert('RGB')

    if size is not None:
        if keep_asp:
            # We need to also scale the height by the same aspect as the width.
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
            print("Resized:", img)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    # We need to permute the dimensions of our image to put channels first.
    img = np.array(img).transpose([2, 0, 1])
    # Convert numpy array to a Torch tensor.
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        # Limit the pixel values between 0 and 255.
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    return img


def tensor_save_bgrimage(tensor, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    return tensor_save_rgbimage(tensor, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch
