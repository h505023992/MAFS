import os
import torch
from models import Autoformer,SMamba,xLSTMMixer, TimesNet, DLinear,  Informer,  PatchTST, Crossformer,  iTransformer, TiDE, TimeMixer,Agent_iTrans_Cooperation,SegRNN




class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'TimeMixer': TimeMixer,
            'iTrans_Cooperation':Agent_iTrans_Cooperation,
            'SegRNN':SegRNN,
            'xLSTMMixer':xLSTMMixer,
            'SMamba':SMamba
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
