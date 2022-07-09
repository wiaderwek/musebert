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
import random


def generate_song(curriculum: Curriculum, emotion: int=1,
                   model_path: Union[None, str]=None, target_length: int=32):
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

    for i in range(0, 5):
        data_in = np.zeros([1, target_length, 8])
        data_in[0, :, 0] = emotion

        for j in range(0, target_length):
            data_in[0, j, 1] = random.randint(0, 9)
            data_in[0, j, 2] = random.randint(0, 7)
            data_in[0, j, 3] = random.randint(0, 7)
            data_in[0, j, 4] = random.randint(0, 3)
            data_in[0, j, 5] = random.randint(0, 12)
            data_in[0, j, 6] = random.randint(0, 5)
            data_in[0, j, 7] = random.randint(0, 8)

        # data_in[0, :, 0] = emotion
        # data_in[0, :, 1] = 9
        # data_in[0, :, 2] = 7
        # data_in[0, :, 3] = 7
        # data_in[0, :, 4] = 3
        # data_in[0, :, 5] = 12
        # data_in[0, :, 6] = 5
        # data_in[0, :, 7] = 8

        #rel_mat = np.random.randint(0, 5, (1, 4, target_length, target_length))
        rel_mat = np.full((1, 4, target_length, target_length), 4)

        mask = np.full((1, 1, 1), 1)

        data_in = torch.from_numpy(data_in.astype(np.int64)).to(device)
        rel_mat = torch.from_numpy(rel_mat.astype(np.int64)).to(device)
        mask = torch.from_numpy(mask.astype(np.int8)).to(device)


        output = model.inference(None, data_in, rel_mat, mask, None, target_length, False)


        result = get_result_atr_mat(output, target_length)
        nmat = decode_atr_mat_to_emotion_nmat(result[0])
        notes = format_convert.nmat_to_notes(nmat, bpm=120, begin=0.0)
        format_convert.save_as_midi_file(notes, "result_2022-05-31_Q{emotion}-{num}".format(emotion=emotion, num=i))



def get_result_atr_mat(output, target_length):
    # output shape [1, 32, 120]
    result = np.zeros((1, target_length, 8), dtype=np.int64)

    for i in range(target_length):
        for j in range(0, 120, 15):
            result[0, i, int(j/15)] = np.argmax(output.detach().numpy()[0, i, j:j+15])

    return result




if __name__ == '__main__':
    # pre-training MuseBERT
    #argmax co 15 to index - wartość
    generate_song(curriculum=all_curriculum, emotion=1, model_path="D:/Tomek/PW-informatyka/Magisterka/Magisterka/MUSEBERT/result_2022-05-26_192723/models/musebert_valid.pt", target_length=128)
