import torch
import torch.nn as nn
import ast

from layers import *


def get_edge_feature(point_cloud, knn_idx, K=20):
    """Construct edge feature for each point
      Args:
        point_cloud: (B, C, N)
        nn_idx: (B, N, K)
        K: int
      Returns:
        edge feat: (B, C, N, K)
    """
    B, C, N = point_cloud.size()
    central_feat = point_cloud.unsqueeze(-1).expand(-1,-1,-1,K)
    knn_idx = knn_idx.unsqueeze(1).expand(-1, C, -1, -1).contiguous().view(B,C,N*K)
    knn_feat = torch.gather(point_cloud, dim=2, index=knn_idx).contiguous().view(B,C,N,K)
    edge_feat = torch.cat((central_feat, knn_feat-central_feat), dim=1)
    return edge_feat


class FeatureEx(nn.Module):
    def __init__(self, num_in_channels, out_channels_list, dropout_list, normalization, activation, norm_momentum):
        super(FeatureEx, self).__init__()

        self.out_channels_list = out_channels_list
        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        previous_out_channels = num_in_channels
        for i,c_out in enumerate(out_channels_list):
            self.mlp_layers.append(Conv1d_wrapper(previous_out_channels, c_out, normalization, activation, norm_momentum))
            self.dropout_layers.append(nn.Dropout(dropout_list[i]))
            previous_out_channels = c_out

    def forward(self, x):
        for l in range(len(self.out_channels_list)):
            x = self.mlp_layers[l](x)
            x = self.dropout_layers[l](x)
        return x


class FeatureEx_residual(nn.Module):
    def __init__(self, num_in_channels, out_channels_list, dropout_list, normalization, activation, norm_momentum):
        super(FeatureEx_residual, self).__init__()

        self.out_channels_list = out_channels_list
        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        previous_out_channels = num_in_channels
        for i,c_out in enumerate(out_channels_list):
            self.mlp_layers.append(Conv1d_wrapper(previous_out_channels, c_out, normalization, activation, norm_momentum))
            self.dropout_layers.append(nn.Dropout(dropout_list[i]))
            previous_out_channels = c_out
        # Use the method B in "Deep Residual Learning for Image Recognition" to solve the dimension mismatch problem between F(x) and x.
        if out_channels_list[0] != out_channels_list[-1]:
            ##Implement Linear layer with Learned Bias and Batch normalization
            self.linear = Conv1d_wrapper(out_channels_list[0], out_channels_list[-1], normalization, None, norm_momentum)

    def forward(self, x):
        for l in range(len(self.out_channels_list)):
            if l == 0:
                x0 = self.mlp_layers[l](x)
                x = x0
            else:
                x = self.mlp_layers[l](x)
            x = self.dropout_layers[l](x)
        if x0.shape[1] != x.shape[1]:
            x0 = self.linear(x0)
        return  x0 + x


class FeatureEx2d(nn.Module):
    def __init__(self, num_in_channels, out_channels_list, kernel_size, dropout_list, normalization, activation, norm_momentum):
        super(FeatureEx2d, self).__init__()

        self.out_channels_list = out_channels_list
        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        previous_out_channels = num_in_channels
        for i,c_out in enumerate(out_channels_list):
            self.mlp_layers.append(Conv2d_wrapper(previous_out_channels, c_out, kernel_size, normalization, activation, norm_momentum))
            self.dropout_layers.append(nn.Dropout2d(dropout_list[i]))
            previous_out_channels = c_out

    def forward(self, x):
        for l in range(len(self.out_channels_list)):
            x = self.mlp_layers[l](x)
            x = self.dropout_layers[l](x)
        return x


class FeatureEx2d_residual(nn.Module):
    def __init__(self, num_in_channels, out_channels_list, kernel_size, dropout_list, normalization, activation, norm_momentum):
        super(FeatureEx2d_residual, self).__init__()

        self.out_channels_list = out_channels_list
        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        previous_out_channels = num_in_channels
        for i,c_out in enumerate(out_channels_list):
            self.mlp_layers.append(Conv2d_wrapper(previous_out_channels, c_out, kernel_size, normalization, activation, norm_momentum))
            self.dropout_layers.append(nn.Dropout2d(dropout_list[i]))
            previous_out_channels = c_out
        # Use the method B in "Deep Residual Learning for Image Recognition" to solve the dimension mismatch problem between F(x) and x.
        if out_channels_list[0] != out_channels_list[-1]:
            self.linear = Conv2d_wrapper(out_channels_list[0], out_channels_list[-1], kernel_size, normalization, None, norm_momentum)

    def forward(self, x):
        for l in range(len(self.out_channels_list)):
            if l == 0:
                x0 = self.mlp_layers[l](x)
                x = x0
            else:
                x = self.mlp_layers[l](x)
            x = self.dropout_layers[l](x)
        if x0.shape[1] != x.shape[1]:
            x0 = self.linear(x0)
        return  x0 + x


class Encoder(nn.Module):
    def __init__(self, opt, in_channels, out_channels_list, dropout_list, k, num_clusters, output_dim, add_residual=True):
        super(Encoder, self).__init__()
        self.opt = opt

        if add_residual:
            self.featEX = FeatureEx2d_residual(in_channels*2, out_channels_list[:-1], (1,1), dropout_list, opt.normalization, opt.activation, opt.norm_momentum)
        else:
            self.featEX = FeatureEx2d(in_channels*2, out_channels_list[:-1], (1,1), dropout_list, opt.normalization, opt.activation, opt.norm_momentum)

        self.K = k
        self.max_pool = nn.MaxPool2d((1, k), stride=(1, k))
        self.avg_pool = nn.AvgPool2d((1, k), stride=(1, k))

        self.reduce_featDim = Conv1d_wrapper(out_channels_list[-2]*2, out_channels_list[-1], opt.normalization, opt.activation, opt.norm_momentum)

        nvlad_input_dim = out_channels_list[-1]
        # Currently it is a Linear layer. If replace 'None' with opt.activation, it becomes FC layer..
        self.netvlad = NetVLAD_wrapper(nvlad_input_dim, num_clusters, output_dim, opt.normalization, None, opt.norm_momentum)


    def forward(self, x, knn_idx):
        """
        :param x (B,C,N)
        :param knn_idx (B,N,K_max)
        """
        N = x.size(2)
        if self.K < knn_idx.size(2):
            knn_idx = knn_idx[:,:,:self.K]
        edge_feat = get_edge_feature(x, knn_idx, self.K)

        local_feat = self.featEX(edge_feat)

        mp_feat = self.max_pool(local_feat).squeeze(3)
        ap_feat = self.avg_pool(local_feat).squeeze(3)

        feat = self.reduce_featDim(torch.cat((mp_feat, ap_feat), dim=1))

        global_feat = self.netvlad(feat) 
        global_feat = global_feat.expand(-1,-1,N)

        feat = torch.cat((feat, global_feat), dim=1)

        return feat


class Encoder_woNetVLAD(nn.Module):
    def __init__(self, opt, in_channels, out_channels_list, dropout_list, k, add_residual=True):
        super(Encoder_woNetVLAD, self).__init__()
        self.opt = opt

        if add_residual:
            self.featEX = FeatureEx2d_residual(in_channels*2, out_channels_list[:-1], (1,1), dropout_list, opt.normalization, opt.activation, opt.norm_momentum)
        else:
            self.featEX = FeatureEx2d(in_channels*2, out_channels_list[:-1], (1,1), dropout_list, opt.normalization, opt.activation, opt.norm_momentum)

        self.K = k
        self.max_pool = nn.MaxPool2d((1, k), stride=(1, k))
        self.avg_pool = nn.AvgPool2d((1, k), stride=(1, k))

        self.reduce_featDim = Conv1d_wrapper(out_channels_list[-2]*2, out_channels_list[-1], opt.normalization, opt.activation, opt.norm_momentum)

        self.global_pool = nn.MaxPool1d(opt.num_point)

    def forward(self, x, knn_idx):
        """
        :param x (B,C,N)
        :param knn_idx (B,N,K_max)
        """
        N = x.size(2)
        if self.K < knn_idx.size(2):
            knn_idx = knn_idx[:,:,:self.K]
        edge_feat = get_edge_feature(x, knn_idx, self.K)

        local_feat = self.featEX(edge_feat)

        mp_feat = self.max_pool(local_feat).squeeze(3)
        ap_feat = self.avg_pool(local_feat).squeeze(3)

        feat = self.reduce_featDim(torch.cat((mp_feat, ap_feat), dim=1))

        global_feat = self.global_pool(feat)
        global_feat = global_feat.expand(-1,-1,N)

        feat = torch.cat((feat, global_feat), dim=1)

        return feat


class Encoder_woEdgeConv(nn.Module):
    def __init__(self, opt, in_channels, out_channels_list, dropout_list, k, num_clusters, output_dim, add_residual=True):
        super(Encoder_woEdgeConv, self).__init__()
        self.opt = opt

        if add_residual:
            self.featEX = FeatureEx_residual(in_channels, out_channels_list, dropout_list, opt.normalization, opt.activation, opt.norm_momentum)
        else:
            self.featEX = FeatureEx(in_channels, out_channels_list, dropout_list, opt.normalization, opt.activation, opt.norm_momentum)

        self.K = k
        self.pool = nn.MaxPool2d((1, k), stride=(1, k))

        nvlad_input_dim = out_channels_list[-1]*2
        self.netvlad = NetVLAD_wrapper(nvlad_input_dim, num_clusters, output_dim, opt.normalization, None, opt.norm_momentum)


    def forward(self, x, knn_idx):
        """
        :param x (B,C,N)
        :param knn_idx (B,N,K_max)
        """
        point_feat = self.featEX(x)

        B, C, N= point_feat.size()
        if self.K < knn_idx.size(2):
            knn_idx = knn_idx[:,:,:self.K]
        knn_idx = knn_idx.unsqueeze(1).expand(-1, C, -1, -1).contiguous().view(B,C,N*self.K)
        knn_feat = torch.gather(point_feat, dim=2, index=knn_idx).contiguous().view(B,C,N,self.K)
        local_feat = self.pool(knn_feat).squeeze(3)
        feat = torch.cat((point_feat, local_feat), dim=1)

        global_feat = self.netvlad(feat) 
        global_feat = global_feat.expand(-1,-1,N)

        feat = torch.cat((feat, global_feat), dim=1)

        return feat


class Encoder_woLocal(nn.Module):
    def __init__(self, opt, in_channels, out_channels_list, dropout_list, k, num_clusters, output_dim, add_residual=True):
        super(Encoder_woLocal, self).__init__()
        self.opt = opt

        if add_residual:
            self.featEX = FeatureEx_residual(in_channels, out_channels_list, dropout_list, opt.normalization, opt.activation, opt.norm_momentum)
        else:
            self.featEX = FeatureEx(in_channels, out_channels_list, dropout_list, opt.normalization, opt.activation, opt.norm_momentum)

        self.K = k
        self.pool = nn.MaxPool2d((1, k), stride=(1, k))

        nvlad_input_dim = out_channels_list[-1]
        self.netvlad = NetVLAD_wrapper(nvlad_input_dim, num_clusters, output_dim, opt.normalization, None, opt.norm_momentum)

    def forward(self, x):
        """
        :param x (B,C,N)
        :param knn_idx (B,N,K_max)
        """
        point_feat = self.featEX(x)
        global_feat = self.netvlad(point_feat)
        global_feat = global_feat.expand(-1,-1,x.shape[-1])
        feat = torch.cat((point_feat, global_feat), dim=1)
        return feat



class Decoder(nn.Module):
    def __init__(self, opt, in_channels, out_channels_list, dropout_list):
        super(Decoder, self).__init__()

        self.out_channels_list = out_channels_list
        self.mlp_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list)-1:
                self.mlp_layers.append(Conv1d_wrapper(previous_out_channels, c_out, opt.normalization, opt.activation, opt.norm_momentum))
                self.dropout_layers.append(nn.Dropout(dropout_list[i]))
            else:
                self.mlp_layers.append(Conv1d_wrapper(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

    def forward(self, x):
        for l in range(len(self.out_channels_list)-1):
            x = self.mlp_layers[l](x)
            x = self.dropout_layers[l](x)
        x = self.mlp_layers[len(self.out_channels_list)-1](x)
        return x


#======================================== Model Variants =================================================
class PS2Net(nn.Module):
    def __init__(self, opt):
        super(PS2Net, self).__init__()
        self.opt = opt
        encoder_conv_paras = ast.literal_eval(opt.encoder_paras)
        encoder_dp_paras = ast.literal_eval(opt.encoder_dropout)
        decoder_conv_paras = ast.literal_eval(opt.decoder_paras)
        decoder_conv_paras.append(opt.classes)
        decoder_dp_paras = ast.literal_eval(opt.decoder_dropout)
        num_clusters_paras = ast.literal_eval(opt.num_clusters)
        output_dim_paras = ast.literal_eval(opt.output_dim)
        knn_list = ast.literal_eval(opt.knn_list)

        decoder_in_channels = 0
        self.num_encoders = len(encoder_conv_paras)
        self.stacked_encoders = nn.ModuleList()
        for i in range(self.num_encoders):
            if i == 0:
                self.stacked_encoders.append(Encoder(opt, opt.input_feat, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            else:
                in_channels = encoder_conv_paras[i-1][-1] + output_dim_paras[i-1]
                self.stacked_encoders.append(Encoder(opt, in_channels, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            decoder_in_channels += encoder_conv_paras[i][-1] + output_dim_paras[i]
        self.decoder = Decoder(opt, decoder_in_channels, decoder_conv_paras, decoder_dp_paras)

    def forward(self, x, knn_idx):
        outputs = []
        for i in range(self.num_encoders):
            x = self.stacked_encoders[i](x, knn_idx)
            outputs.append(x)
        output = torch.cat(outputs, dim=1)
        y = self.decoder(output)

        return y


class PS2Net_woNetVLAD(nn.Module):
    def __init__(self, opt):
        super(PS2Net_woNetVLAD, self).__init__()
        self.opt = opt
        encoder_conv_paras = ast.literal_eval(opt.encoder_paras)
        encoder_dp_paras = ast.literal_eval(opt.encoder_dropout)
        decoder_conv_paras = ast.literal_eval(opt.decoder_paras)
        decoder_conv_paras.append(opt.classes)
        decoder_dp_paras = ast.literal_eval(opt.decoder_dropout)
        num_clusters_paras = ast.literal_eval(opt.num_clusters)
        output_dim_paras = ast.literal_eval(opt.output_dim)
        knn_list = ast.literal_eval(opt.knn_list)

        decoder_in_channels = 0
        self.num_encoders = len(encoder_conv_paras)
        self.stacked_encoders = nn.ModuleList()
        for i in range(self.num_encoders):
            if i == 0:
                self.stacked_encoders.append(Encoder_woNetVLAD(opt, opt.input_feat, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i]))
            else:
                in_channels = encoder_conv_paras[i-1][-1] + output_dim_paras[i-1]
                self.stacked_encoders.append(Encoder_woNetVLAD(opt, in_channels, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i]))
            decoder_in_channels += encoder_conv_paras[i][-1] + output_dim_paras[i]
        self.decoder = Decoder(opt, decoder_in_channels, decoder_conv_paras, decoder_dp_paras)

    def forward(self, x, knn_idx):
        outputs = []
        for i in range(self.num_encoders):
            x = self.stacked_encoders[i](x, knn_idx)
            outputs.append(x)
        output = torch.cat(outputs, dim=1)
        y = self.decoder(output)

        return y


class PS2Net_woEdgeConv(nn.Module):
    def __init__(self, opt):
        super(PS2Net_woEdgeConv, self).__init__()
        self.opt = opt
        encoder_conv_paras = ast.literal_eval(opt.encoder_paras)
        encoder_dp_paras = ast.literal_eval(opt.encoder_dropout)
        decoder_conv_paras = ast.literal_eval(opt.decoder_paras)
        decoder_conv_paras.append(opt.classes)
        decoder_dp_paras = ast.literal_eval(opt.decoder_dropout)
        num_clusters_paras = ast.literal_eval(opt.num_clusters)
        output_dim_paras = ast.literal_eval(opt.output_dim)
        knn_list = ast.literal_eval(opt.knn_list)

        decoder_in_channels = 0
        self.num_encoders = len(encoder_conv_paras)
        self.stacked_encoders = nn.ModuleList()
        for i in range(self.num_encoders):
            if i == 0:
                self.stacked_encoders.append(Encoder_woEdgeConv(opt, opt.input_feat, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            else:
                in_channels = encoder_conv_paras[i-1][-1]*2 + output_dim_paras[i-1]
                self.stacked_encoders.append(Encoder_woEdgeConv(opt, in_channels, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            decoder_in_channels += encoder_conv_paras[i][-1]*2 + output_dim_paras[i]
        self.decoder = Decoder(opt, decoder_in_channels, decoder_conv_paras, decoder_dp_paras)

    def forward(self, x, knn_idx):
        outputs = []
        for i in range(self.num_encoders):
            x = self.stacked_encoders[i](x, knn_idx)
            outputs.append(x)
        output = torch.cat(outputs, dim=1)
        y = self.decoder(output)

        return y


class PS2Net_woLocal(nn.Module):
    def __init__(self, opt):
        super(PS2Net_woLocal, self).__init__()
        self.opt = opt
        encoder_conv_paras = ast.literal_eval(opt.encoder_paras)
        encoder_dp_paras = ast.literal_eval(opt.encoder_dropout)
        decoder_conv_paras = ast.literal_eval(opt.decoder_paras)
        decoder_conv_paras.append(opt.classes)
        decoder_dp_paras = ast.literal_eval(opt.decoder_dropout)
        num_clusters_paras = ast.literal_eval(opt.num_clusters)
        output_dim_paras = ast.literal_eval(opt.output_dim)
        knn_list = ast.literal_eval(opt.knn_list)

        decoder_in_channels = 0
        self.num_encoders = len(encoder_conv_paras)
        self.stacked_encoders = nn.ModuleList()
        for i in range(self.num_encoders):
            if i == 0:
                self.stacked_encoders.append(Encoder_woLocal(opt, opt.input_feat, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            else:
                in_channels = encoder_conv_paras[i-1][-1] + output_dim_paras[i-1]
                self.stacked_encoders.append(Encoder_woLocal(opt, in_channels, encoder_conv_paras[i], encoder_dp_paras[i], knn_list[i], num_clusters_paras[i], output_dim_paras[i]))
            decoder_in_channels += encoder_conv_paras[i][-1] + output_dim_paras[i]
        self.decoder = Decoder(opt, decoder_in_channels, decoder_conv_paras, decoder_dp_paras)

    def forward(self, x, knn_idx):
        outputs = []
        for i in range(self.num_encoders):
            x = self.stacked_encoders[i](x)
            outputs.append(x)
        output = torch.cat(outputs, dim=1)
        y = self.decoder(output)

        return y