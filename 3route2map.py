import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ThreeRoute2Map(nn.Module):
    def __init__(self, cfg: dict, n_classes = 1000, init_weights = True):
        super(ThreeRoute2Map, self).__init__()

        self.threeroute = ThreeRoute(cfg)
        self.leftwing = nn.Sequential(
            ToMap(cfg, 0),
            ToMap(cfg, 1),
            ToMap(cfg, 2),
            ToMap(cfg, 3),
            ToMap(cfg, 4, True)
        )
        self.rightwing = nn.Sequential(
            ToMap(cfg, 0),
            ToMap(cfg, 1),
            ToMap(cfg, 2),
            ToMap(cfg, 3),
            ToMap(cfg, 4, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def concat_classifier(self, x1, x2):
        out = torch.cat([x1, x2], 1)
        #print(out.shape)
        out = self.classifier(out)
        return out
        

    def forward(self, x):
        return self.sequencing(x)

    def sequencing(self, x):
        out = self.threeroute(x)
        out_left = self.leftwing(out)
        out_right = self.rightwing(out)
        return self.concat_classifier(out_left, out_right)
    

class ThreeRoute(nn.Module):
    def __init__(self, cfg):
        super(ThreeRoute, self).__init__()

        self.sequences = []
        self.links = []
        self.make_links()
        self.make_sequences()

        self.sq1 = self.sequences[0]
        self.sq2 = self.sequences[1]
        self.sq3 = self.sequences[2]

        self.w1 = self.links[0]
        self.w2 = self.links[1]
        self.w3 = self.links[2]
        self.w4 = self.links[3]
        self.w5 = self.links[4]
        self.w6 = self.links[5]

    def make_links(self):
        for i in range(6):
            self.links.append(nn.Parameter(torch.tensor([0.33], dtype=torch.float), requires_grad=True))

    def make_sequences(self):
        channels = [1, 3, 5]
        for channel in channels:
            self.sequences.append(
                nn.Sequential(
                    nn.Conv2d(3, cfg['channel'][0], channel),
                    nn.BatchNorm2d(cfg['channel'][0]),
                    nn.ReLU(),
            ))

    def forward(self, x):
        sq_outs = []
        for sq in self.sequences:
            sq_outs.append(sq(x))

        min_size = min([sq_outs[0].size(-1), sq_outs[1].size(-1), sq_outs[2].size(-1)])

        id = nn.AdaptiveMaxPool2d((min_size, min_size))(x)

        pool_outs = []
        for sq in sq_outs:
            pool_outs.append(nn.AdaptiveMaxPool2d((min_size, min_size))(sq))

        out1 = torch.cat([pool_outs[0] * self.links[0], pool_outs[1] * self.links[1]], 1)#1x3 => 1
        out2 = torch.cat([pool_outs[2] * self.links[2], pool_outs[0] * self.links[3]], 1)#1x5 => 3
        out3 = torch.cat([pool_outs[1] * self.links[4], pool_outs[2] * self.links[5]], 1)#3x5 => 5
        #print(out1.shape) #out_channel * 2

        return [out1, out2, out3], id

        
        

class ToMap(nn.Module):
    def __init__(self, cfg, index, cat = False):
        super(ToMap, self).__init__()
        
        self.cat = cat
        self.sequences = []
        self.links = []
        self.make_links()
        self.make_sequences(cfg, index)
        self.cfg = cfg
        self.index = index
        
        if index < 1:
            self.id_conv = nn.Conv2d(cfg['channel'][index] * 2, cfg['channel'][index + 1], 1)
        else:
            self.id_conv = nn.Conv2d(cfg['channel'][index] * 3, cfg['channel'][index + 1], 1)

        if self.cat:
            self.fc = nn.Sequential(
                nn.Linear(10240, 2048),
                nn.ReLU()
            )

        self.sq1 = self.sequences[0]
        self.sq2 = self.sequences[1]
        self.sq3 = self.sequences[2]

        self.w1 = self.links[0]
        self.w2 = self.links[1]
        self.w3 = self.links[2]
        self.w4 = self.links[3]
        self.w5 = self.links[4]
        self.w6 = self.links[5]
        self.w7 = self.links[6]
        self.w8 = self.links[7]
        self.w9 = self.links[8]

    def make_links(self):
        for i in range(9):
            self.links.append(nn.Parameter(torch.tensor([0.33], dtype=torch.float), requires_grad=True))

    def make_sequences(self, cfg, index):
        channels = [1, 3, 5]
        for channel in channels:
            if index < 1:
                self.sequences.append(
                    nn.Sequential(
                        nn.Conv2d(cfg['channel'][index] * 2, cfg['channel'][index + 1], channel),
                        nn.BatchNorm2d(cfg['channel'][index + 1]),
                        nn.ReLU(),
                ))
            else:
                self.sequences.append(
                    nn.Sequential(
                        nn.Conv2d(cfg['channel'][index] * 3, cfg['channel'][index + 1], channel),
                        nn.BatchNorm2d(cfg['channel'][index + 1]),
                        nn.ReLU(),
                ))

    def forward(self, x):
        x = x[0]
        id = x[1]
        sq_outs = []
        for i, _x in enumerate(x):
            sq_outs.append(self.sequences[i](_x))

        #print(sq_outs[0].size(-1), sq_outs[1].size(-1), sq_outs[2].size(-1))
        min_size = min([sq_outs[0].size(-1), sq_outs[1].size(-1), sq_outs[2].size(-1)])

        id = nn.AdaptiveMaxPool2d((min_size, min_size))(id)
        id = self.id_conv(id)
        

        pool_outs = []
        for sq in sq_outs:
            pool_outs.append(nn.AdaptiveMaxPool2d((min_size, min_size))(sq))

        #print(pool_outs[0].shape, pool_outs[1].shape, pool_outs[2].shape, id.shape)

        out1 = torch.cat([pool_outs[0] * self.links[0], pool_outs[1] * self.links[1], id * self.links[6]], 1)#1x3 => 1
        out2 = torch.cat([pool_outs[2] * self.links[2], pool_outs[0] * self.links[3], id * self.links[7]], 1)#1x5 => 3
        out3 = torch.cat([pool_outs[1] * self.links[4], pool_outs[2] * self.links[5], id * self.links[8]], 1)#3x5 => 5
        #print(out3.shape)#cfg[index + 1] in_channel * 3
        if self.cat:
            out = torch.cat([out1, out2, out3, id], 1)
            out = out.view(-1, out.size(1) * min_size * min_size)
            #print(out.shape)
            out = self.fc(out)
            #print(out.shape)
            return out
        else:
            return [out1, out2, out3], id


cfg = {
    'channel': [16, 32, 64, 64, 32, 16]
}


def threeRoute2map(n_classes):
    return ThreeRoute2Map(cfg, n_classes, init_weights=True)