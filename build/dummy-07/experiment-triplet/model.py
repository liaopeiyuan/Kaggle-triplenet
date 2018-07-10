from common import *




## block ## -------------------------------------

class Linear_Bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_Bn, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels,bias=False)
        torch.nn.init.xavier_uniform(self.linear.weight)
        self.bn   = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


## net  ## -------------------------------------
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        num_points = 3


        self.feature = nn.Sequential(
            
            Linear_Bn(3*num_points, 32),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn( 32,  64),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn( 64,  128),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn(128,  128),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn(128,  236),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),
            
            Linear_Bn(236,  246),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn(246, 256),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),
            

            Linear_Bn(256, 246),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn(246, 236),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),
            
            Linear_Bn(236,  128),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),
            
            Linear_Bn(128,  128),
            nn.PReLU(),nn.Dropout(p=0.5, inplace=True),
            
            Linear_Bn(128,   64),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

            Linear_Bn(64,   32),
            nn.PReLU(), nn.Dropout(p=0.5, inplace=True),

        )

        self.logit = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # self.target = nn.Sequential(
        #     Linear_Bn(64, 3)
        # )


    def forward(self, x):

        batch_size  = x.size(0)
        x = x.view(batch_size,-1)

        x      = self.feature(x)
        logit  = self.logit(x).view(-1)

        return logit


    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


### run ##############################################################################


def run_check_net():

    #create dummy data
    batch_size  = 5
    num_points  = 100
    tracklet = torch.randn((batch_size,num_points,3))
    tracklet = tracklet.cuda()

    net = TripletNet().cuda()
    logit = net(tracklet)

    # print(type(net))
    # print(net,'\n')

    print(logit,'\n')
    print(logit.size(),'\n')


    print('')





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_net()

    print( 'sucessful!')