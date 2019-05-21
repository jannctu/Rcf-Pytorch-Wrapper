import torch
import numpy as np
import h5py


def init_model(model,weight_path='models/bsds_converted.h5'):
    w = h5py.File(weight_path, 'r')
    model.conv1_1.weight.data = torch.from_numpy(np.array(w['conv1_1.weight']))
    model.conv1_1.bias.data = torch.from_numpy(np.array(w['conv1_1.bias']))
    model.conv1_1_down.weight.data = torch.from_numpy(np.array(w['conv1_1_down.weight']))
    model.conv1_1_down.bias.data = torch.from_numpy(np.array(w['conv1_1_down.bias']))

    model.conv1_2.weight.data = torch.from_numpy(np.array(w['conv1_2.weight']))
    model.conv1_2.bias.data = torch.from_numpy(np.array(w['conv1_2.bias']))
    model.conv1_2_down.weight.data = torch.from_numpy(np.array(w['conv1_2_down.weight']))
    model.conv1_2_down.bias.data = torch.from_numpy(np.array(w['conv1_2_down.bias']))

    ## BLOCK 2
    model.conv2_1.weight.data = torch.from_numpy(np.array(w['conv2_1.weight']))
    model.conv2_1.bias.data = torch.from_numpy(np.array(w['conv2_1.bias']))
    model.conv2_1_down.weight.data = torch.from_numpy(np.array(w['conv2_1_down.weight']))
    model.conv2_1_down.bias.data = torch.from_numpy(np.array(w['conv2_1_down.bias']))

    model.conv2_2.weight.data = torch.from_numpy(np.array(w['conv2_2.weight']))
    model.conv2_2.bias.data = torch.from_numpy(np.array(w['conv2_2.bias']))
    model.conv2_2_down.weight.data = torch.from_numpy(np.array(w['conv2_2_down.weight']))
    model.conv2_2_down.bias.data = torch.from_numpy(np.array(w['conv2_2_down.bias']))

    ## BLOCK 3
    model.conv3_1.weight.data = torch.from_numpy(np.array(w['conv3_1.weight']))
    model.conv3_1.bias.data = torch.from_numpy(np.array(w['conv3_1.bias']))
    model.conv3_1_down.weight.data = torch.from_numpy(np.array(w['conv3_1_down.weight']))
    model.conv3_1_down.bias.data = torch.from_numpy(np.array(w['conv3_1_down.bias']))

    model.conv3_2.weight.data = torch.from_numpy(np.array(w['conv3_2.weight']))
    model.conv3_2.bias.data = torch.from_numpy(np.array(w['conv3_2.bias']))
    model.conv3_2_down.weight.data = torch.from_numpy(np.array(w['conv3_2_down.weight']))
    model.conv3_2_down.bias.data = torch.from_numpy(np.array(w['conv3_2_down.bias']))

    model.conv3_3.weight.data = torch.from_numpy(np.array(w['conv3_3.weight']))
    model.conv3_3.bias.data = torch.from_numpy(np.array(w['conv3_3.bias']))
    model.conv3_3_down.weight.data = torch.from_numpy(np.array(w['conv3_3_down.weight']))
    model.conv3_3_down.bias.data = torch.from_numpy(np.array(w['conv3_3_down.bias']))

    ## BLOCK 4
    model.conv4_1.weight.data = torch.from_numpy(np.array(w['conv4_1.weight']))
    model.conv4_1.bias.data = torch.from_numpy(np.array(w['conv4_1.bias']))
    model.conv4_1_down.weight.data = torch.from_numpy(np.array(w['conv4_1_down.weight']))
    model.conv4_1_down.bias.data = torch.from_numpy(np.array(w['conv4_1_down.bias']))

    model.conv4_2.weight.data = torch.from_numpy(np.array(w['conv4_2.weight']))
    model.conv4_2.bias.data = torch.from_numpy(np.array(w['conv4_2.bias']))
    model.conv4_2_down.weight.data = torch.from_numpy(np.array(w['conv4_2_down.weight']))
    model.conv4_2_down.bias.data = torch.from_numpy(np.array(w['conv4_2_down.bias']))

    model.conv4_3.weight.data = torch.from_numpy(np.array(w['conv4_3.weight']))
    model.conv4_3.bias.data = torch.from_numpy(np.array(w['conv4_3.bias']))
    model.conv4_3_down.weight.data = torch.from_numpy(np.array(w['conv4_3_down.weight']))
    model.conv4_3_down.bias.data = torch.from_numpy(np.array(w['conv4_3_down.bias']))

    ## BLOCK 5
    model.conv5_1.weight.data = torch.from_numpy(np.array(w['conv5_1.weight']))
    model.conv5_1.bias.data = torch.from_numpy(np.array(w['conv5_1.bias']))
    model.conv5_1_down.weight.data = torch.from_numpy(np.array(w['conv5_1_down.weight']))
    model.conv5_1_down.bias.data = torch.from_numpy(np.array(w['conv5_1_down.bias']))

    model.conv5_2.weight.data = torch.from_numpy(np.array(w['conv5_2.weight']))
    model.conv5_2.bias.data = torch.from_numpy(np.array(w['conv5_2.bias']))
    model.conv5_2_down.weight.data = torch.from_numpy(np.array(w['conv5_2_down.weight']))
    model.conv5_2_down.bias.data = torch.from_numpy(np.array(w['conv5_2_down.bias']))

    model.conv5_3.weight.data = torch.from_numpy(np.array(w['conv5_3.weight']))
    model.conv5_3.bias.data = torch.from_numpy(np.array(w['conv5_3.bias']))
    model.conv5_3_down.weight.data = torch.from_numpy(np.array(w['conv5_3_down.weight']))
    model.conv5_3_down.bias.data = torch.from_numpy(np.array(w['conv5_3_down.bias']))

    model.score_dsn1.weight.data = torch.from_numpy(np.array(w['score-dsn1.weight']))
    model.score_dsn1.bias.data = torch.from_numpy(np.array(w['score-dsn1.bias']))
    model.score_dsn2.weight.data = torch.from_numpy(np.array(w['score-dsn2.weight']))
    model.score_dsn2.bias.data = torch.from_numpy(np.array(w['score-dsn2.bias']))
    model.score_dsn3.weight.data = torch.from_numpy(np.array(w['score-dsn3.weight']))
    model.score_dsn3.bias.data = torch.from_numpy(np.array(w['score-dsn3.bias']))
    model.score_dsn4.weight.data = torch.from_numpy(np.array(w['score-dsn4.weight']))
    model.score_dsn4.bias.data = torch.from_numpy(np.array(w['score-dsn4.bias']))
    model.score_dsn5.weight.data = torch.from_numpy(np.array(w['score-dsn5.weight']))
    model.score_dsn5.bias.data = torch.from_numpy(np.array(w['score-dsn5.bias']))

    model.score_final.weight.data = torch.from_numpy(np.array(w['new-score-weighting.weight']))
    model.score_final.bias.data = torch.from_numpy(np.array(w['new-score-weighting.bias']))


