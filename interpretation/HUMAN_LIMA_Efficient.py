import math
import random
import numpy as np

from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
# import torchvision.transforms as transforms

from itertools import combinations
from collections import OrderedDict

class HumanLIMA(object):
    def __init__(self, 
                 model,
                 lambda1 = 2.0,    # consistency
                 lambda2 = 1.0,     # colla.
                 batch_size = 32,
                 threshold = 0.9,
                 softmax=False,
                 search_scope = 8,
                 pending_samples = 8,
                 update_step = 20,
                 ):
        
        self.model = model
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        try:
            self.device = next(model.parameters()).device
        except:
            self.device = self.model.model.device

        self.batch_size = batch_size
        self.threshold = threshold
        
        self.softmax = softmax
        
        self.k = 10
        
        self.search_scope = search_scope
        self.update_step = update_step
        
        self.pending_samples = pending_samples

    def proccess_compute_consistency_score(self, batch_input_images):
        """
        Compute each consistency score
        按 self.batch_size 划分，计算 target_label 和 counter_label 的分数
        """
        with torch.no_grad(): 
            self.predicted_scores = self.model(batch_input_images)
            if self.softmax:
                self.predicted_scores = F.softmax(self.predicted_scores, dim=1)
            consistency_scores = self.predicted_scores[:, self.target_label]
        return consistency_scores

    def evaluation_minimal_changeset(self, S_set):
        """
        Given a subset, return a best sample index
        """
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)

        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)
        alpha_batch = alpha_batch.expand(-1, -1, -1, 3)
        
        if len(S_set) == 0 or self.update_count % self.update_step == 0:
            # Positive samples search
            source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)
            
        else:
            alpha_batch = alpha_batch[:self.search_scope]
            # Positive samples search with scope
            source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)

        # source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)
        batch_input_images = alpha_batch * source_tensor   # 扰动最少的区域       # torch.Size([51, 1365, 2048, 3])
        batch_input_images_reverse = (1 - alpha_batch) * source_tensor    # 扰动最少的区域的取反,即暴露最少的区域
        
        batch_input_images = batch_input_images.permute(0, 3, 1, 2)
        batch_input_images_reverse = batch_input_images_reverse.permute(0, 3, 1, 2)

        # Del尽可能少的区域，使决策正类置信度下降，反类置信度上升；
        # Insert尽可能少的区域，使决策正类置信度上升，反类置信度下降；
        with torch.no_grad():
            score_consistency = self.proccess_compute_consistency_score(batch_input_images)
            score_collaboration = 1 - self.proccess_compute_consistency_score(batch_input_images_reverse)

        # Overall submodular score
        smdl_scores = self.lambda1 * score_consistency + self.lambda2 * score_collaboration
        arg_max_index = smdl_scores.argmax().cpu().item()
        
        if len(S_set) == 0 or self.update_count % self.update_step == 0:
            indices = torch.argsort(smdl_scores, descending=True)
            sorted_V = [self.V_set[i] for i in indices]
            self.V_set = sorted_V
            
            # Update 0 -> Have been sorted
            S_set.append(self.V_set[0])
            self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[0]).float().to(self.device)
            del self.V_set[0]
            
        else:
            # Update 0 -> Have been sorted
            S_set.append(self.V_set[arg_max_index])
            self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
            del self.V_set[arg_max_index]

        # Update
        # S_set.append(self.V_set[arg_max_index])
        # self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
        # del self.V_set[arg_max_index]
        
        # self.saved_json_file["confidence_score"].append(score_confidence[arg_max_index].cpu().item())
        self.saved_json_file["insertion_score"].append(score_consistency[arg_max_index].cpu().item())
        self.saved_json_file["deletion_score"].append(1 - score_collaboration[arg_max_index].cpu().item())
        self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())

        return S_set
    
    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["insertion_score"] = []
        self.saved_json_file["deletion_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2

    def get_merge_set(self):
        # define a subset
        S_set = []
        # self.refer_baseline = np.zeros_like(self.V_set[0]).astype(np.float32)
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)

        self.update_count = 0
        for i in range(len(self.V_set)):
            S_set = self.evaluation_minimal_changeset(S_set)
            if self.saved_json_file["insertion_score"][-1] > self.threshold:
                break
            if i == self.k:
                break
            self.update_count += 1
        return S_set
    
    def __call__(self, source_tensor, V_set, gt_id = None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
            V_set (_type_): (n, h, w, 1)
        """        
        self.save_file_init()
        self.source_tensor = source_tensor.clone().detach().permute(1, 2, 0)

        self.target_label = gt_id
        # self.counter_label = counter_id
        
        # self.region_area = image.shape[0] * image.shape[1]

        self.V_set = V_set.copy()
        
        S_set = self.get_merge_set()
        
        return S_set, self.saved_json_file

