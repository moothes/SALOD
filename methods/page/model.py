import torch
from torch import nn
from torch.nn import functional as F

def up2_size(x, x_size):
    x = F.interpolate(x, size=x_size, mode='bilinear')
    return x

def gen_conv(In, Out):
    yield nn.Conv2d(In, Out, 5, padding=2)
    yield nn.ReLU(inplace=True)
    yield nn.Conv2d(Out, Out, 5, padding=2)
    yield nn.Sigmoid()

class Atten(nn.Module):
    def __init__(self, channel, scale=1):
        super(Atten, self).__init__()
        
        self.scale = scale
        if scale == 1:
            self.conv = nn.Conv2d(channel, 1, 1)
        else:
            self.conv = nn.Conv2d(channel, 1, 1, dilation=3)

    def forward(self, x):
        x_size = x.size()
        
        x = self.conv(x).relu_()
        if self.scale > 1:
            x = F.max_pool2d(x, kernel_size=(self.scale, self.scale), stride=(self.scale, self.scale))
        #x = F.softmax(F.interpolate(x, size=x_size[-2:], mode='bilinear').view(x_size[0], -1), dim=1)
        x = F.interpolate(x, size=x_size[-2:], mode='bilinear')
        
        #x_size[1] = 1
        return x.view(x_size[0], 1, x_size[2], x_size[3])
    
class decoder(nn.Module):
    def __init__(self, feat):
        super(decoder, self).__init__()
        
        new_feat = [64, 128, 256, 256, 512, 512]
        
        self.sal_out = [nn.Sequential(*list(gen_conv(feat[i], new_feat[i]))).cuda() for i in range(6)]
        self.edge_out = [nn.Sequential(*list(gen_conv(feat[i], new_feat[i]))).cuda() for i in range(5)]
        
        self.sal6_conv = nn.Conv2d(new_feat[5], 1, 1)
        
        self.edge5_conv = nn.Conv2d(new_feat[4], 1, 1)
        self.sal5_conv = nn.Conv2d(new_feat[4] + 1, 256, 3, padding=1)
        self.atten5a = Atten(256, 1) #nn.Conv2d(256, 1, 1)
        self.atten5b = Atten(256, 2) #nn.Conv2d(256, 1, 1, dilation=3)
        #self.atten5b_mp = nn.MaxUnpool2d(2, 2)
        self.sal5_out = nn.Conv2d(257, 256, 3, padding=1)
        self.sal5_up = nn.Conv2d(256, 1, 1)
        
        self.edge4_conv_1 = nn.Conv2d(new_feat[3] + 1, 128, 3, padding=1)
        self.edge4_conv = nn.Conv2d(128, 1, 1)
        
        self.sal4_conv_1 = nn.Conv2d(new_feat[3] + 2, 128, 3, padding=1)
        self.atten4a = Atten(128, 1)
        self.atten4b = Atten(128, 2)
        self.atten4c = Atten(128, 4)
        self.sal4_out = nn.Conv2d(129, 128, 3, padding=1)
        self.sal4_up = nn.Conv2d(128, 1, 1)
        
        self.edge3_conv_1 = nn.Conv2d(new_feat[2] + 2, 128, 3, padding=1)
        self.edge3_conv = nn.Conv2d(128, 1, 1)
        
        self.sal3_conv_1 = nn.Conv2d(new_feat[2] + 3, 128, 3, padding=1)
        self.atten3a = Atten(128, 1)
        self.atten3b = Atten(128, 2)
        self.atten3c = Atten(128, 4)
        self.atten3d = Atten(128, 8)
        self.sal3_out = nn.Conv2d(129, 128, 3, padding=1)
        self.sal3_up = nn.Conv2d(128, 1, 1)
        
        self.edge2_conv_1 = nn.Conv2d(new_feat[1] + 3, 64, 3, padding=1)
        self.edge2_conv = nn.Conv2d(64, 1, 1)
        
        self.sal2_conv_1 = nn.Conv2d(new_feat[1] + 4, 64, 3, padding=1)
        self.atten2a = Atten(64, 1)
        self.atten2b = Atten(64, 2)
        self.atten2c = Atten(64, 4)
        self.atten2d = Atten(64, 8)
        self.atten2e = Atten(64, 16)
        self.sal2_out = nn.Conv2d(65, 64, 3, padding=1)
        self.sal2_up = nn.Conv2d(64, 1, 1)
        
        self.edge1_conv_1 = nn.Conv2d(new_feat[0] + 4, 32, 3, padding=1)
        self.edge1_conv = nn.Conv2d(32, 1, 1)
        
        self.sal1_conv_1 = nn.Conv2d(new_feat[0] + 5, 32, 3, padding=1)
        self.atten1a = Atten(32, 1)
        self.atten1b = Atten(32, 2)
        self.atten1c = Atten(32, 4)
        self.atten1d = Atten(32, 8)
        self.atten1e = Atten(32, 16)
        self.atten1f = Atten(32, 32)
        self.sal1_out = nn.Conv2d(33, 32, 3, padding=1)
        self.sal1_up = nn.Conv2d(32, 1, 1)

    def forward(self, xs, x0_size, phase='test'):
        batch = x0_size[0]
        x_size = x0_size[-2:]
        
        sal_xs = [self.sal_out[i](xs[i]) for i in range(6)]
        edge_xs = [self.edge_out[i](xs[i]) for i in range(5)]
        
        # 6666666666666666666666666666666
        sal6      = self.sal6_conv(sal_xs[5]).relu_()
        
        # 555555555555555555555555
        s5_size   = sal_xs[4].size()[-2:]
        edge5     = self.edge5_conv(edge_xs[4])
        
        sal_t5    = F.interpolate(sal6, size=s5_size, mode='bilinear')
        sal5_out  = self.sal5_conv(torch.cat([sal_xs[4], sal_t5], dim=1)).relu_()
        
        atten5a   = self.atten5a(sal5_out)
        atten5b   = self.atten5b(sal5_out)
        atten5    = (atten5a + atten5b) / 2.
        sal5_out  = sal5_out * (atten5 + 1)
        sal5_out  = torch.cat([sal5_out, edge5], dim=1)
        sal5_out  = self.sal5_out(sal5_out).relu_()
        sal5      = self.sal5_up(sal5_out)
        
        # 4444444444444444444444444444444444444
        s4_size   = sal_xs[3].size()[-2:]
        edge_t4   = F.interpolate(edge5, size=s4_size, mode='bilinear')
        edge4     = self.edge4_conv_1(torch.cat([edge_xs[3], edge_t4], dim=1)).relu_()
        edge4     = self.edge4_conv(edge4)
        
        sal_t41   = F.interpolate(sal6, size=s4_size, mode='bilinear')
        sal_t42   = F.interpolate(sal5, size=s4_size, mode='bilinear')
        sal4_out  = torch.cat([sal_xs[3], sal_t41, sal_t42], dim=1)
        sal4_out  = self.sal4_conv_1(sal4_out).relu_()
        
        atten4a   = self.atten4a(sal4_out)
        atten4b   = self.atten4b(sal4_out)
        atten4c   = self.atten4c(sal4_out)
        atten4    = (atten4a + atten4b +atten4c) / 3.
        sal4_out  = sal4_out * (atten4 + 1)
        sal4_out  = torch.cat([sal4_out, edge4], dim=1)
        sal4_out  = self.sal4_out(sal4_out).relu_()
        sal4      = self.sal4_up(sal4_out)
        
        # 3333333333333333333333333333333333333
        s3_size   = sal_xs[2].size()[-2:]
        edge_t31  = F.interpolate(edge5, size=s3_size, mode='bilinear')
        edge_t32  = F.interpolate(edge4, size=s3_size, mode='bilinear')
        edge3     = self.edge3_conv_1(torch.cat([edge_xs[2], edge_t31, edge_t32], dim=1)).relu_()
        edge3     = self.edge3_conv(edge3)
        
        sal_t31   = F.interpolate(sal6, size=s3_size, mode='bilinear')
        sal_t32   = F.interpolate(sal5, size=s3_size, mode='bilinear')
        sal_t33   = F.interpolate(sal4, size=s3_size, mode='bilinear')
        sal3_out  = torch.cat([sal_xs[2], sal_t31, sal_t32, sal_t33], dim=1)
        sal3_out  = self.sal3_conv_1(sal3_out).relu_()
        
        atten3a   = self.atten3a(sal3_out)
        atten3b   = self.atten3b(sal3_out)
        atten3c   = self.atten3c(sal3_out)
        atten3d   = self.atten3d(sal3_out)
        atten3    = (atten3a + atten3b +atten3c + atten3d) / 4.
        sal3_out  = sal3_out * (atten3 + 1)
        sal3_out  = torch.cat([sal3_out, edge3], dim=1)
        sal3_out  = self.sal3_out(sal3_out).relu_()
        sal3      = self.sal3_up(sal3_out)
        
        # 222222222222222222222222222222222
        s2_size   = sal_xs[1].size()[-2:]
        edge_t21  = F.interpolate(edge5, size=s2_size, mode='bilinear')
        edge_t22  = F.interpolate(edge4, size=s2_size, mode='bilinear')
        edge_t23  = F.interpolate(edge3, size=s2_size, mode='bilinear')
        edge2     = self.edge2_conv_1(torch.cat([edge_xs[1], edge_t21, edge_t22, edge_t23], dim=1)).relu_()
        edge2     = self.edge2_conv(edge2)
        
        sal_t21   = F.interpolate(sal6, size=s2_size, mode='bilinear')
        sal_t22   = F.interpolate(sal5, size=s2_size, mode='bilinear')
        sal_t23   = F.interpolate(sal4, size=s2_size, mode='bilinear')
        sal_t24   = F.interpolate(sal3, size=s2_size, mode='bilinear')
        sal2_out  = torch.cat([sal_xs[1], sal_t21, sal_t22, sal_t23, sal_t24], dim=1)
        sal2_out  = self.sal2_conv_1(sal2_out).relu_()
        
        atten2a   = self.atten2a(sal2_out)
        atten2b   = self.atten2b(sal2_out)
        atten2c   = self.atten2c(sal2_out)
        atten2d   = self.atten2d(sal2_out)
        atten2e   = self.atten2e(sal2_out)
        atten2    = (atten2a + atten2b +atten2c + atten2d + atten2e) / 5.
        sal2_out  = sal2_out * (atten2 + 1)
        sal2_out  = torch.cat([sal2_out, edge2], dim=1)
        sal2_out  = self.sal2_out(sal2_out).relu_()
        sal2      = self.sal2_up(sal2_out)
        
        # 11111111111111111111111111111111
        s1_size   = sal_xs[0].size()[-2:]
        edge_t11  = F.interpolate(edge5, size=s1_size, mode='bilinear')
        edge_t12  = F.interpolate(edge4, size=s1_size, mode='bilinear')
        edge_t13  = F.interpolate(edge3, size=s1_size, mode='bilinear')
        edge_t14  = F.interpolate(edge2, size=s1_size, mode='bilinear')
        edge1     = self.edge1_conv_1(torch.cat([edge_xs[0], edge_t11, edge_t12, edge_t13, edge_t14], dim=1)).relu_()
        edge1     = self.edge1_conv(edge1)
        
        sal_t11   = F.interpolate(sal6, size=s1_size, mode='bilinear')
        sal_t12   = F.interpolate(sal5, size=s1_size, mode='bilinear')
        sal_t13   = F.interpolate(sal4, size=s1_size, mode='bilinear')
        sal_t14   = F.interpolate(sal3, size=s1_size, mode='bilinear')
        sal_t15   = F.interpolate(sal2, size=s1_size, mode='bilinear')
        sal1_out  = torch.cat([sal_xs[0], sal_t11, sal_t12, sal_t13, sal_t14, sal_t15], dim=1)
        sal1_out  = self.sal1_conv_1(sal1_out).relu_()
        
        atten1a   = self.atten1a(sal1_out)
        atten1b   = self.atten1b(sal1_out)
        atten1c   = self.atten1c(sal1_out)
        atten1d   = self.atten1d(sal1_out)
        atten1e   = self.atten1e(sal1_out)
        atten1f   = self.atten1f(sal1_out)
        atten1    = (atten1a + atten1b +atten1c + atten1d + atten1e + atten1f) / 6.
        #sal1_out  = sal1_out * (atten1 + 1)
        sal1_out  = sal1_out * atten1
        sal1_out  = torch.cat([sal1_out, edge1], dim=1)
        sal1_out  = self.sal1_out(sal1_out).relu_()
        sal1      = self.sal1_up(sal1_out)
        
        edges = [edge5, edge4, edge3, edge2, edge1]
        sals = [sal6, sal5, sal4, sal3, sal2, sal1]
        
        #for e in edges:
        #    print(e.size())
        #for s in sals:
        #    print(s.size())
        
        out_dict = {}
        out_dict['edge'] = [up2_size(edge, x_size) for edge in edges]
        out_dict['sal'] = [up2_size(sal, x_size) for sal in sals]
        out_dict['final'] = out_dict['sal'][-1]
        return out_dict
        
        
class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        
        self.encoder = encoder
        self.maxp = nn.MaxUnpool2d(2, 2)
        feat.append(feat[-1])
        
        self.decoder = decoder(feat)
        
    def forward(self, x, phase='test'):
        x_size = x.size()
    
        xs = self.encoder(x)
        x = F.max_pool2d(xs[-1], kernel_size=(2, 2), stride=(2, 2))
        #x = self.maxp(xs[-1])
        xs.append(x)
        
        out = self.decoder(xs, x_size)
        
        return out

    