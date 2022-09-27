from __future__ import division
import os
import torch
from torchvision import transforms
import argparse
import json
import torch.distributed as dist
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("MacOSX")

from utils import is_image_file, pil_loader, process_images, normalize_tensor
from HRNet import HRNet_dropout, HRNet_var

def load_img(img_path):
    print('> Input image file: ', img_path)
    if is_image_file(img_path):
        img = pil_loader(img_path)
    else:
        RuntimeError('> Image format is not supported')
    return img

def preprocess(img, input_res, norm_mean, norm_std, device):
    print('> Preprocess the image')
    input_transforms = transforms.Compose(
        [
            transforms.Resize(input_res),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]
    )
    processed_img = input_transforms(img)
    processed_img = processed_img.unsqueeze(dim=0).to(device)
    return processed_img

def load_model(model_file, hypes, device):
    print('> Load the pretrained model. Model type:', hypes['arch']['config'])

    if hypes['arch']['config'] == 'HRNet_do':
        seg_model = HRNet_dropout(config=hypes).to(device)
        bayes_method = 'dropout'
    elif hypes['arch']['config'] == 'HRNet_var':
        seg_model = HRNet_var(config=hypes)
        bayes_method = 'variational'

    if hypes['arch']['config'][:5] == 'HRNet':
        pretrained_dict = torch.load(model_file, map_location=device)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        prefix = "module."
        keys = sorted(pretrained_dict.keys())
        for key in keys:
            if key.startswith(prefix):
                newkey = key[len(prefix):]
                pretrained_dict[newkey] = pretrained_dict.pop(key)
        # also strip the prefix in metadata if any.
        if "_metadata" in pretrained_dict:
            metadata = pretrained_dict["_metadata"]
            for key in list(metadata.keys()):
                if len(key) == 0:
                    continue
                newkey = key[len(prefix):]
                metadata[newkey] = metadata.pop(key)

        seg_model.load_state_dict(pretrained_dict)
        seg_model.to(device)

    else:
        seg_model.load_state_dict(torch.load(model_file, map_location=device))
    
    weights_factor = hypes['data']['class_weights']
    
    return seg_model, bayes_method, weights_factor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference on images for Capsule Segmentation (SegCaps)")
    parser.add_argument('--model', type=str, help='Path to directory with model file and hypes.json [required]')
    parser.add_argument('--image', type=str, help='Path to image to run inference on [required]')
    parser.add_argument('--gt', type=str, help='Optional path to ground truth file, will return confusion matrix.')
    parser.add_argument('--target', type=int, default=1, help='Optional target class, default to 0.')
    parser.add_argument('--n_MC', type=int, default=1, help='Optional number of times to run the image, default 16.')
    parser.add_argument('--out_res', nargs='+', type=int, default=None, help='Optional output resolution')
    parser.add_argument('--thresh', type=float, default=0.75, help='Optional threshold')
    parser.add_argument('--factor', type=float, default=None, help='Optional factor')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hypes_file = os.path.join(args.model, 'hypes.json')
    with open(hypes_file,'r') as f:
        hypes = json.load(f)

    model_file = hypes['model']
    if args.out_res is not None:
        if len(args.out_res) == 1:
            input_res = [args.out_res[0], args.out_res[0]]
        elif len(args.out_res) > 2:
            print('out res must be length 2')
            exit()
        else:
            input_res = args.out_res
    else:
        input_res = hypes['arch']['image_shape'][1:3]
    num_classes = hypes['arch']['num_classes']
    class_colors = hypes['data']['class_colours']
    class_labels = hypes['data']['class_labels']
    image_shape = hypes['arch']['image_shape']
    batch_size = hypes['solver']['batch_size']
    batch_shape = [batch_size] + image_shape

    orig_img = load_img(args.image)
    resized_img = preprocess(orig_img, input_res, hypes['data']['pop_mean'], hypes['data']['pop_std0'], device)

    seg_model, bayes_method, weights_factor = load_model(model_file, hypes, device)

    with torch.no_grad():
        seg_model.train(False)

        out = []
        var = []
        for j in range(args.n_MC):
            with torch.no_grad():
                outDict = seg_model(resized_img)
                out.append(outDict['out'].squeeze().detach())
                var.append(outDict['logVar'].squeeze().detach())
            print(' >> ' * j + 'X' + '<' * (args.n_MC - j - 1), end="\r", flush=True)

        out = torch.stack(out)
        var = torch.stack(var)
        varmax = var.max()
        varmin = var.min()
        out = normalize_tensor(out)
        var = normalize_tensor(var) * (varmax - varmin)

        savename = os.path.join(os.getcwd(), 'outputs',
                                str(hypes['arch']['config']), str(args.thresh),
                                str(bayes_method + '_' + os.path.splitext(os.path.basename(args.image))[0]))

        os.makedirs(os.path.dirname(savename), mode=0o755, exist_ok=True)
        fscore = process_images(hypes, savename, orig_img, out, var, args.gt, input_res,
                                threshold=args.thresh, printout=True)
