import numpy as np
from config import PARAMETERS
import torch
from tools.model import MinkUNeXt
import MinkowskiEngine as ME


def calculate_global_descriptor(pcd, weights = PARAMETERS.weights_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinkUNeXt(in_channels=1, out_channels=512, D=3).cuda()
    model.eval()
    model.load_state_dict(
        torch.load(weights, map_location=device))
    with torch.no_grad():
        points = np.asarray(pcd.points)
        coords = ME.utils.sparse_quantize(coordinates=points,
                                          quantization_size=PARAMETERS.qsize).cuda()
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).cuda()
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
        descriptor = model(batch)
        d = descriptor.cpu().detach().numpy()
    return d[0]


def calculate_local_descriptors(pcd, layer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptors = []
    model = MinkUNeXt(in_channels=1, out_channels=512, D=3, extract_local=True).cuda()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if layer == "conv0p1s1":
        model.conv0p1s.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bn0":
        model.bn0.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu1":
        model.relu1.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "conv1p1s2":
        model.conv1p1s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bn1":
        model.bn1.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu2":
        model.relu2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block1":
        model.block1.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "conv2p2s2":
        model.conv2p2s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bn2":
        model.bn2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu3":
        model.relu3.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block2":
        model.block2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "conv3p4s2":
        model.conv3p4s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bn3":
        model.bn3.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu4":
        model.relu4.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block3":
        model.block3.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "conv4p8s2":
        model.conv4p8s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bn4":
        model.bn4.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu5":
        model.relu5.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block4":
        model.block4.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "convtr4p16s2":
        model.convtr4p16s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bntr4":
        model.bntr4.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu6":
        model.relu6.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block5":
        model.block5.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "convtr5p8s2":
        model.convtr5p8s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bntr5":
        model.bntr5.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu7":
        model.relu7.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block6":
        model.block6.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "convtr6p4s2":
        model.convtr6p4s2.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "bntr6":
        model.bntr6.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "relu8":
        model.relu8.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "block7":
        model.block7.register_forward_hook(get_activation('local_descriptors'))
    elif layer == "final":
        model.final.register_forward_hook(get_activation('local_descriptors'))

    model.eval()

    model.load_state_dict(
        torch.load(PARAMETERS.weights_directory, map_location=device))

    with torch.no_grad():

        points = np.asarray(pcd.points)
        coords = ME.utils.sparse_quantize(coordinates=points,
                                          quantization_size=PARAMETERS.qsize).cuda()

        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).cuda()
        tfield = ME.TensorField(coordinates=bcoords, features=feats,
                                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                device=device)
        descriptor = model(tfield.sparse())
        local_descriptors = activation['local_descriptors']
        coordinates = local_descriptors.coordinates.cpu().detach().numpy()
        local_features = local_descriptors.features.cpu().detach().numpy()
        d = descriptor.cpu().detach().numpy()
        descriptors.append(d[0])

    return np.asarray(coordinates), np.array(local_features),