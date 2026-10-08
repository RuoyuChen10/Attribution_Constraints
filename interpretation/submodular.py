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

class BlackBoxSingleModalCounterfactualSubModularExplanation(object):
    def __init__(self, 
                 model,
                 k = 10,
                 lambda1 = 20.0,    # consistency
                 lambda2 = 5.0,     # colla.
                 batch_size = 16,
                 counter_confidence = 0.4,
                 device = "cuda"):
        self.k = k
        
        self.model = model
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.device = device

        self.batch_size = batch_size

        self.counter_confidence = counter_confidence

    def proccess_compute_consistency_score(self, batch_input_images):
        """
        Compute each consistency score
        按 self.batch_size 划分，计算 target_label 和 counter_label 的分数
        """
        all_consistency_scores = []
        all_counter_consistency_scores = []

        with torch.no_grad():
            for i in range(0, batch_input_images.size(0), self.batch_size):
                batch = batch_input_images[i:i + self.batch_size]

                # 模型输出 softmax 概率
                predicted_scores = torch.softmax(self.model(batch), dim=-1)

                # 取 top-2 类别
                _, top2_indices = torch.topk(predicted_scores, k=2, dim=1)

                # 第一名是 target label
                target_labels = top2_indices[:, 0]
                # 第二名是 counter label
                counter_labels = top2_indices[:, 1]

                # 按照预测的类别取对应概率
                consistency_scores = predicted_scores[torch.arange(predicted_scores.size(0)), target_labels]
                counter_consistency_scores = predicted_scores[torch.arange(predicted_scores.size(0)), counter_labels]

                all_consistency_scores.append(consistency_scores)
                all_counter_consistency_scores.append(counter_consistency_scores)

        # 拼接所有 batch 的结果
        all_consistency_scores = torch.cat(all_consistency_scores, dim=0)
        all_counter_consistency_scores = torch.cat(all_counter_consistency_scores, dim=0)

        return all_consistency_scores, all_counter_consistency_scores

    def evaluation_minimal_changeset(self, S_set):
        """
        Given a subset, return a best sample index
        """
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)

        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)
        alpha_batch = alpha_batch.expand(-1, -1, -1, 3)

        source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)
        batch_input_images = (1 - alpha_batch) * source_tensor   # 扰动最少的区域       # torch.Size([51, 1365, 2048, 3])
        batch_input_images_reverse = alpha_batch * source_tensor    # 扰动最少的区域的取反,即暴露最少的区域
        
        batch_input_images = batch_input_images.permute(0, 3, 1, 2)
        batch_input_images_reverse = batch_input_images_reverse.permute(0, 3, 1, 2)

        # Del尽可能少的区域，使决策正类置信度下降，反类置信度上升；
        # Insert尽可能少的区域，使决策正类置信度上升，反类置信度下降；
        with torch.no_grad():
            del_gt, del_counter = self.proccess_compute_consistency_score(batch_input_images)
            insert_gt, insert_counter = self.proccess_compute_consistency_score(batch_input_images_reverse)

        # Overall submodular score
        smdl_scores = self.lambda1 * del_counter + self.lambda1 * (1 - insert_counter) + self.lambda2 * (1 - del_gt) + self.lambda2 * insert_gt
        arg_max_index = smdl_scores.argmax().cpu().item()

        # Update
        S_set.append(self.V_set[arg_max_index])
        self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
        del self.V_set[arg_max_index]

        self.confidence_judge = del_counter[arg_max_index].item()
        if self.confidence_judge > self.max_confidence:
            self.max_confidence = self.confidence_judge
            
            self.aug_data = batch_input_images[arg_max_index]

        return S_set

    def get_merge_set(self):
        # define a subset
        S_set = []
        # self.refer_baseline = np.zeros_like(self.V_set[0]).astype(np.float32)
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)
        
        self.confidence_judge = 0
        self.max_confidence = 0
        self.aug_data= None

        for i in range(self.k):
            S_set = self.evaluation_minimal_changeset(S_set)
            if self.confidence_judge > self.counter_confidence:
                break
        
        return S_set
    
    def __call__(self, source_tensor, V_set, gt_id = None, counter_id = None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
            V_set (_type_): (n, h, w, 1)
        """        
        self.source_tensor = source_tensor.clone().detach().permute(1, 2, 0)

        self.target_label = gt_id
        self.counter_label = counter_id
        
        # self.region_area = image.shape[0] * image.shape[1]

        self.V_set = V_set.copy()
        
        self.get_merge_set()
        
        return self.aug_data, self.max_confidence

