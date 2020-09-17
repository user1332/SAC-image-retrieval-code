import os
import torch
import shutil
import json
import math
from PIL import Image
from joblib import Parallel, delayed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return


class Ranker():
    def __init__(self, root, image_split_file, transform=None, num_workers=16):
        self.num_workers = num_workers
        self.root = root
        with open(image_split_file, 'r') as f:
            data = json.load(f)
        self.data = data
        self.ids = range(len(self.data))
        self.transform = transform
        return

    def get_item(self, index):
        data = self.data
        id = self.ids[index]
        img_name = data[id] + '.jpg'
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, data[id]

    def get_items(self, indexes):
        items = Parallel(n_jobs=1)(
            delayed(self.get_item)(
                i) for i in indexes)
        images, meta_info = zip(*items)
        images = torch.stack(images, dim=0)
        return images, meta_info

    def update_emb(self, image_encoder, batch_size=64):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor([i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images)
            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor([i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images)
            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin
        
    def update_emb_bert(self, image_encoder, batch_size=64):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor([i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images,0)
            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor([i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images,0)
            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin

#         print(torch.equal(self.data_emb[0], self.data_emb[1]))
#         print((self.data_emb[4]-self.data_emb[5]).sum())
#         print((self.data_emb[1]-self.data_emb[2]).sum())
#         print((self.data_emb[6]-self.data_emb[5]).sum())
#         print((self.data_emb[7]-self.data_emb[8]).sum())
        

    def compute_rank(self, inputs, target_ids,candidate_ids):
        rankings = []
        total = inputs.size(0)
        sum2 = 0
        sum = 0
        sum3= 0
        for i in range(total):
            distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
            distances[self.data_asin.index(candidate_ids[i])]= 1e10
            ranking = (distances < distances[self.data_asin.index(target_ids[i])]).sum(0)
            if ranking < 1:
                sum = sum + 1
            if ranking < 10:
                sum2 = sum2 +1
            if ranking < 50:
                sum3 = sum3 +1
            
        return sum,sum2, sum3, total

    def get_nearest_neighbors(self, inputs, target_id,topK=10):
        neighbors = []
        sum = 0
        tot_sum = 0
        for i in range(inputs.size(0)):
            tot_sum = tot_sum + 1
            [_, neighbor] = (self.data_emb - inputs[i]).pow(2).sum(dim=1).topk(dim=0, k=topK, largest=False, sorted=True)
#             print([self.data_asin[index] for index in neighbor], target_id[i])
            neighbors.append(neighbor)
            if self.data_asin.index(target_id[i]) in neighbor:
                sum= sum + 1
#         print(neighbor, )
        return sum, tot_sum