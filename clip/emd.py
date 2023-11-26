import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_with_flow

class EMD_Aligner(nn.Module):
    def __init__(self, visual_layers: int, text_layers: int, visual_width: int, text_width: int,
                 use_att: bool = True, use_hid: bool = True):
        super(EMD_Aligner, self).__init__()
        
        self.visual_layers, self.text_layers = visual_layers, text_layers
        self.visual_width, self.text_width = visual_width, text_width
        self.use_att, self.use_hid = use_att, use_hid
        
        self.att_visual_weight = np.ones(visual_layers) / visual_layers
        self.hid_visual_weight = np.ones(visual_layers) / visual_layers
        
        self.att_text_weight = np.ones(text_layers) / text_layers
        self.hid_text_weight = np.ones(text_layers) / text_layers
        
        self.proj = nn.Linear(text_width, visual_width, bias=False)
        
    def get_new_layer_weight(self, trans_matrix, distance_matrix, type_update: str = "hid"):
        
        if type_update == "att":
            visual_layer_weight = np.copy(self.att_visual_weight)
            text_layer_weight = np.copy(self.att_text_weight)
        elif type_update == "hid":
            visual_layer_weight = np.copy(self.hid_visual_weight)
            text_layer_weight = np.copy(self.hid_text_weight)
        else:
            raise NotImplementedError
            
        distance_matrix = distance_matrix.detach().cpy().numpy().astype('float64')
        trans_weight = np.sum(trans_matrix * distance_matrix, axis=-1)
        
        for i in range(self.visual_layers):
            visual_layer_weight[i] = trans_weight[i] / visual_layer_weight[i]
        weight_sum = np.sum(visual_layer_weight)
        for i in range(self.visual_layers):
            if visual_layer_weight[i] != 0: visual_layer_weight[i] = weight_sum / visual_layer_weight[i]
                
        trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, axis=-1)
        for i in range(self.text_layers):
            text_layer_weight[i] = trans_weight[i + self.visual_layers] / text_layer_weight[i]
        weight_sum = np.sum(text_layer_weight)
        for i in range(self.text_layers):
            if text_layer_weight[i] != 0: text_layer_weight[i] = weight_sum / text_layer_weight[i]
                
        visual_layer_weight = visual_layer_weight / np.sum(visual_layer_weight)
        text_layer_weight = text_layer_weight / np.sum(text_layer_weight)
        
        if type_update == "att":
            self.att_visual_weight = visual_layer_weight
            self.att_text_weight = text_layer_weight
        elif type_update == "hid":
            self.hid_visual_weight = visual_layer_weight
            self.hid_text_weight = text_layer_weight
        else:
            raise NotImplementedError
            
    def forward(self, visual_atts, text_atts, visual_hids, text_hids, device):
        if self.use_att:
            att_loss, att_trans_matrix, att_distance_matrix = self.emd_att_loss(visual_atts, text_atts)
            
            self.get_new_layer_weight(att_trans_matrix, att_distance_matrix, 'att')
            att_loss = att_loss.to(visual_atts.device)
        
        if self.use_hid:
            hid_loss, hid_trans_matrix, hid_distance_matrix = self.emd_hid_loss(visual_hids, text_hids)
            
            self.get_new_layer_weight(hid_trans_matrix, hid_distance_matrix, 'hid')
            hid_loss = hid_loss.to(visual_hids.device)
            
        if self.use_att and self.use_hid and not self.separate:
            visual_weight = np.mean(np.stack([self.att_visual_weight, self.hid_visual_weight]), 0)
            text_weight = np.mean(np.stack([self.att_text_weight, self.hid_text_weight]), 0)
            self.att_visual_weight, self.hid_visual_weight = visual_weight, visual_weight
            self.att_text_weight, self.hid_text_weight = text_weight, text_weight
        
        return att_loss, hid_loss
    
    def emd_att_loss(self, visual_atts, text_atts):
        visual_layer_weight = np.concatenate((self.att_visual_weight, np.zeros(self.text_layers)))
        text_layer_weight = np.concatenate((np.zeros(self.visual_layers), self.att_text_weight))
        total_num = self.visual_layers + self.text_layers
        distance_matrix = torch.zeros([total_num, total_num]).cuda()
        
        for i in range(self.visual_layers):
            visual_att = visual_atts[i]
            visual_att = torch.where(visual_att <= 1e-2, torch.zeros_like(visual_att).to(device), visual_att)
            for j in range(self.text_layers):
                text_att = text_atts[j]
                text_att = torch.where(text_att <= 1e-2, torch.zeros_like(text_att).to(device), text_att)
                tmp_loss = self.loss_mse(visual_att, text_att)
                distance_matrix[i][j + self.visual_layers] = distance_matrix[j + self.visual_layers][i] = tmp_loss
        
        _, trans_matrix = emd_with_flow(visual_layer_weight, text_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix
    
    def emd_hid_loss(self, visual_hids, text_hids):
        visual_layer_weight = np.concatenate((self.hid_visual_weight, np.zeros(self.text_layers)))
        text_layer_weight = np.concatenate((np.zeros(self.visual_layers), self.hid_text_weight))
        total_num = self.visual_layers + self.text_layers
        distance_matrix = torch.zeros([total_num, total_num]).cuda()
        
        for i in range(self.visual_layers):
            visual_hid = visual_hids[i+1]
            for j in range(self.text_layers):
                text_hid = text_hids[j+1]
                tmp = loss_mse(visual_hid, text_hid)
                distance_matrix[i][j+self.visual_layers] = distance_matrix[j+self.visual_layers][i] = tmp_loss
                
        _, trans_matrix = emd_with_flow(visual_layer_weight, text_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        hid_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return hid_loss, trans_matrix, distance_matrix