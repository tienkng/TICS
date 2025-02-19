import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(config):
    if config.model_kwargs.encoder.pretrained_name == "movinet":
        model = TICsMoviNet(config=config)
    else:
        model = TICsVideoNet(config=config)

    return model


class TICsVideoNet(nn.Module):
    def __init__(
        self,
        config  
    ):
        super().__init__()
        import timm
        
        self.mode = config.model_kwargs.mode
        self.encoder = timm.create_model(
            config.model_kwargs.encoder.pretrained_name,
            pretrained=config.model_kwargs.encoder.pretrained,
            num_classes=0,
        )
        self.set_parameter_requires_grad(self.encoder, config.model_kwargs.encoder.freeze)
        encoder_output = self.encoder.feature_info[-1]["num_chs"] * 4
        
        # Temporal network
        self.temporal = nn.LSTM(
            input_size=encoder_output,
            hidden_size=config.model_kwargs.temporal.lstm_hidden_size,
            dropout=config.model_kwargs.temporal.lstm_dropout,
            num_layers=config.model_kwargs.temporal.lstm_layers,
            batch_first=False,
            bidirectional=False   
        )
        self.fc1 = nn.Linear(
            config.model_kwargs.temporal.lstm_hidden_size,
            config.model_kwargs.temporal.lstm_hidden_size,
        )
        self.fc2 = nn.Linear(
            config.model_kwargs.temporal.lstm_hidden_size,
            config.model_kwargs.num_classes,
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = []
        hidden = None
        
        for frame_idx in range(x.shape[1]):
            # encoder extraction
            features = self.encoder(x[:, frame_idx, :, :, :])
            
            # temporal 
            out, hidden = self.temporal(features, hidden)
            if not self.mode == 'seq_to_one':
                idx_output = self.fc2(F.relu(out))
                output.append(idx_output)
        
        # head layers
        if not self.mode == 'seq_to_one':
            output = torch.stack(output)
            output = output.permute(1, 0, 2)
            
            return output
        
        output = self.fc1(F.relu(out))
        output = self.fc2(output)
        
        return output
    
    def set_parameter_requires_grad(self, model, freeze):
        for param in model.blocks[:freeze].parameters():
            param.requires_grad = False
                
class TICsMoviNet:
    def __init__(self, config):
        
        from src.libs.movinet.movinets.models import MoViNet
        from src.libs.movinet.movinets.config import _C
        
        self.num_classes = config.model_kwargs.num_classes
        self.seq_len = config.model_kwargs.seq_len

        model_name = {
            "movinet_a0" : _C.MODEL.MoViNetA0,
            "movinet_a1" : _C.MODEL.MoViNetA1,
            "movinet_a2" : _C.MODEL.MoViNetA2,
            "movinet_a3" : _C.MODEL.MoViNetA3,
            # "movinet_a4" : _C.MODEL.MoViNetA4,
            # "movinet_a5" : _C.MODEL.MoViNetA5,
        }

        model = MoViNet(
            cfg=model_name[config.model_kwargs.encoder.arch],
            causal=config.model_kwargs.encoder.causal,
            pretrained=config.model_kwargs.encoder.pretrained,
            
        )

        # build head classification
        if config.model_kwargs.mode == 'seq_to_one':
            model.classifier[-1] = torch.nn.Conv3d(
                in_channels = 2048, 
                out_channels = self.num_classes, 
                kernel_size = (1, 1, 1)
            )
        
        elif config.model_kwargs.mode == 'seq_to_seq':
            model.classifier[-1] = nn.Sequential(
                nn.Conv3d(
                    in_channels=2048,
                    out_channels=self.num_classes * self.seq_len,
                    kernel_size = (1,1,1)
                ),
                ClassificationHeads(
                    in_shape = self.num_classes * self.seq_len,
                    out_shape = self.num_classes,
                    seq_len = self.seq_len
                )
            )
            
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClassificationHeads(torch.nn.Module):
    def __init__(self, in_shape, out_shape, seq_len):
        super().__init__()

        self.seq_len = seq_len
        for i in range(self.seq_len):
            net = torch.nn.Linear(in_shape, out_shape)
            setattr(self, f'head_{i}', net)

    def forward(self, x):
        x = x.squeeze()
        output = list()
        for i in range(self.seq_len):
            net = getattr(self, f'head_{i}')
            out = net(x)
            output.append(out)
        output = torch.stack(output)
        output = output.permute(1, 0, 2)
        return output
        

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.utils.config import read_config_from_file
    
    # x_movinet = torch.randint(0, 255, size=(8, 3, 5, 172, 172), dtype=torch.float32).cuda() / 255. # batch_size, channels, sequence len, with, heigh
    
    # config_file = read_config_from_file('src/configs/movinet.yaml')
    # tic_model = build_model(config_file)
    # tic_model = tic_model.model.to(x_movinet.device)
    # predict = tic_model.forward(x_movinet)
    # print("Model: ", predict.shape)
    
    x_mobinet = torch.randint(0, 255, size=(8, 5, 3, 224, 224), dtype=torch.float32).cuda() / 255. # batch_size, channels, sequence len, with, heigh
    
    config_file = read_config_from_file('src/configs/mobilenet.yaml')
    tic_model = build_model(config_file)
    tic_model = tic_model.to(x_mobinet.device)
    predict = tic_model.forward(x_mobinet)
    print("Model: ", predict.shape, "\n", predict)