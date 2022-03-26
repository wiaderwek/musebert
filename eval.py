import torch
from torch import optim
from curricula import music_curriculum, Curriculum, all_curriculum
from amc_dl import format_convert
from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, \
    ConstantScheduler, TrainingInterface
from utils import get_linear_schedule_with_warmup
from typing import Union
from note_attribute_repr import decode_atr_mat_to_emotion_nmat
import numpy as np


def generate_song(curriculum: Curriculum, emotion: int=1,
                   model_path: Union[None, str]=None):
    """
    The main function to train a MuseBERT model.

    :param curriculum: input parameters
    :param model_path: None or pre-trained model path.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create data_loaders and initialize model specified by the curriculum.
    _, model = curriculum(device)

    # load a pre-trained model if necessary.
    if model_path is not None:
        model.load_model(model_path, device)

    # mask dla rel_mat = 4, [1, 4, 4, 32]
    # mask dla atr_mat = (9, 7, 7, 3, 12, 5, 8)
    # data_in = [1, 32, 8]
    # mask = [1, 1, 1]

    data_in = np.zeros([1, 32, 8])
    data_in[0, :, 0] = emotion
    data_in[0, :, 1] = 9
    data_in[0, :, 2] = 7
    data_in[0, :, 3] = 7
    data_in[0, :, 4] = 3
    data_in[0, :, 5] = 12
    data_in[0, :, 6] = 5
    data_in[0, :, 7] = 8

    rel_mat = np.full((1, 4, 4, 32), 4)

    mask = np.full((1, 1, 1), 1)

    output = model.eval(None, data_in, rel_mat, mask, None, 32, False)


    result = get_result_atr_mat(output)
    nmat = decode_atr_mat_to_emotion_nmat(result)
    notes = format_convert.nmat_to_notes(nmat)
    format_convert.save_as_midi_file(notes, "output")



def get_result_atr_mat(output):
    # output shape [1, 32, 120]
    result = np.zeros((1, 32, 8))

    for i in range(32):
        for j in range(0, 120, 15):
            result[0, i, j/15] = np.argmax(output[0, i, j:j+15])

    return result




if __name__ == '__main__':
    # pre-training MuseBERT
    #argmax co 15 to index - wartość
    generate_song(curriculum=all_curriculum, emotion=1, model_path="")
