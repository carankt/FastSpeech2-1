import torch
from utils.hparams import HParam
from dataset.texts import valid_symbols
from fastspeech import FeedForwardTransformer

def test_fastspeech():
    idim = len(valid_symbols)
    hp = HParam("configs/default.yaml")
    hp.train.ngpu = 0
    odim = hp.audio.num_mels
    model = FeedForwardTransformer(idim, odim, hp)
    x = torch.ones(2, 50)
    input_length = torch.tensor([20, 50])
    y = torch.ones(2, 100, 80)
    out_length = torch.tensor([50, 100])
    dur = torch.ones(2, 50)
    e = torch.ones(2, 100)
    p = torch.ones(2, 100)
    loss, report_dict = model(x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda(), dur.cuda(), e.cuda(),
          p.cuda())