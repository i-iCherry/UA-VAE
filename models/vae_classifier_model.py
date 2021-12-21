import torch
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses
from torch.nn import functional as F
from . import confidnet
import numpy as np

class VaeClassifierModel(VaeBasicModel):
    """
    This class implements the VAE classifier model, using the VAE framework with the classification downstream task.
    """

    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the vae regression model
        parser.add_argument('--class_num', type=int, default=0,
                            help='the number of classes for the classification task')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_classifier class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.append('classifier')
        # specify the metrics you want to print out.
        self.metric_names = ['accuracy']
        # input tensor
        self.label = None
        # output tensor
        self.y_out = None

        # confidnet
        self.y_trust = None
        self.loss_tcp = None

        self.y_prob = None
        self.y_ture = None
        self.y_pred = None

        self.savea =[]
        self.saveb =[]

        self.store_mean = []
        self.store_label = []


        # define the network
        self.netDown = networks.define_down(param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, param.class_num, None, None, param.init_type,
                                            param.init_gain, self.gpu_ids)
        # define the classification loss
        self.lossFuncClass = losses.get_loss_func('CE', param.reduction)
        self.loss_classifier = None
        self.metric_accuracy = None

        if self.isTrain:
            # Set the optimizer
            self.optimizer_Down = torch.optim.Adam(self.netDown.parameters(), lr=param.lr, betas=(param.beta1, 0.999),
                                                   weight_decay=param.weight_decay)
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_Down)

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        VaeBasicModel.set_input(self, input_dict)
        self.label = input_dict['label'].to(self.device)

    def forward(self):
        VaeBasicModel.forward(self)

        self.tcpa = self.confidnet_a(self.level_3_A).view(-1, 1)
        self.tcpb = self.confidnet_b(self.level_3_B).view(-1, 1)
        self.tcpc = self.confidnet(self.level_3_C).view(-1, 1)
        self.tcp = torch.cat((self.tcpa, self.tcpb, self.tcpc), 1)

        self.y_out = self.netDown(self.latent)

        with torch.no_grad():
            self.y_prob = F.softmax(self.y_out, dim=1)
            _, self.y_pred = torch.max(self.y_prob, 1)

            index = self.data_index
            self.y_true = self.label
            id_x = torch.tensor(self.y_true).long().view(-1, 1)
            self.y_trust = self.y_prob.gather(1, id_x)
            self.y_trust = torch.cat((self.y_trust, self.y_trust), 1)


        return self.y_out

    def cal_losses(self):
        """Calculate losses"""

        VaeBasicModel.cal_losses(self)

        # Calculate the classification loss (downstream loss)
        self.loss_classifier = self.lossFuncClass(self.y_out, self.label)
        # LOSS DOWN
        self.loss_down = self.loss_classifier

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down  # self.param.k_embed *

    def cal_conf_loss(self):

        criterion = torch.nn.MSELoss()


        self.loss_conf_a = criterion(self.tcpa, self.y_trust)
        self.loss_conf_b = criterion(self.tcpb, self.y_trust)

        self.loss_conf = self.loss_conf_a + self.loss_conf_b

        self.loss_all_2 = self.param.k_embed * self.loss_embed + self.loss_conf

    def update(self):

        VaeBasicModel.update(self)
        if self.phase == 'p4':
            self.forward()
            self.optimizer_confidnet_a.zero_grad()  # Set gradients to zero
            self.optimizer_confidnet_b.zero_grad()


            self.cal_conf_loss()
            self.loss_conf.backward()
            self.optimizer_confidnet_a.step()  # Set gradients to zero
            self.optimizer_confidnet_b.step()

        elif self.phase == 'p5':
            #dropout
            self.param.dropout_p = 0
            self.param.lr = self.param.lr / 10.0
            self.forward()
            self.optimizer_confidnet_a.zero_grad()  # Set gradients to zero
            self.optimizer_confidnet_b.zero_grad()

            self.optimizer_Embed.zero_grad()  # Set gradients to zero
            self.cal_losses()  # Calculate losses
            self.cal_conf_loss()
            self.loss_all_2.backward()  # Backpropagation
            self.optimizer_Embed.step()
            self.optimizer_confidnet_a.step()  # Set gradients to zero
            self.optimizer_confidnet_b.step()

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            index = self.data_index

            return {'index': index, 'y_true': self.y_true, 'y_pred': self.y_pred, 'y_prob': self.y_prob, "mean":self.mean}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        self.metric_accuracy = (output_dict['y_true'] == output_dict['y_pred']).sum().item() / len(
            output_dict['y_true'])
