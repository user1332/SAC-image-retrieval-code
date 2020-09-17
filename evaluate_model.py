import argparse
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms
from data_loader_decoder import get_loader
from build_vocab import Vocabulary
from models import ImageEncoder_MulGate_3_2Seq_SA_multiple_sentence_attention_map_soft_multiple, DummyCaptionEncoder_without_embed_multiple_random_embeddings
from recall import create_exp_dir, Ranker
import torch.nn.functional as F
import numpy as np
torch.manual_seed(67)
# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
import glob
from dominate import document
from dominate.tags import *



def eval_batch(data_loader, image_encoder, caption_encoder, ranker):

    ranker.update_emb(image_encoder)
    rankings = []
    loss = []   
    sum1 = 0
    sum10= 0
    sum50= 0
    total = 0
    for i, (target_images, candidate_images, captions, lengths, meta_info) in enumerate(data_loader):
        with torch.no_grad():
            candidate_images = candidate_images.cuda()
            lengths = torch.LongTensor(lengths).cuda()
            captions = captions.cuda()

            caption_low,caption_mid,caption = caption_encoder(captions,lengths)
            candidate_ft = image_encoder.forward_compose(candidate_images, caption_low,caption_mid, 0)
            out_emb= caption+candidate_ft
            target_asins = [ meta_info[m]['target'] for m in range(len(meta_info)) ]
            candidate_asins = [ meta_info[m]['candidate'] for m in range(len(meta_info)) ]
            temp_sum1, temp_sum_10, temp_sum_50, temp_total = ranker.compute_rank(out_emb, target_asins,candidate_asins)
            sum1 = sum1 + temp_sum1
            sum10 = sum10 + temp_sum_10
            sum50 = sum50 + temp_sum_50
            total = total + temp_total

    print("Recall@1 = ",sum1/total, "Recall@10 = ",sum10/total, "Recall@50 = ",sum50/total)
    
    
    
def evaluate (args):

    data = args.data_set
    print ("Evaluating trained model on ", data)
    img_path = "data/resized_images/" + data
    train_cap = "data/captions/cap." + data + ".train.json"
    val_cap = "data/captions/cap." + data + ".val.json"
    dict_vocab = "data/captions/dict." + data + ".json"
    val_set = "data/image_splits/split." + data + ".val.json"

    transform_dev = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    ranker = Ranker("data/resized_images/" + data, "data/image_splits/split."+args.data_set+".val.json",
                        transform=transform_dev, num_workers=args.num_workers)

    vocab = Vocabulary()

    vocab.load(dict_vocab)



    data_loader_dev = get_loader(img_path,
                                 val_cap,
                                 vocab, transform_dev,
                                 args.batch_size, shuffle=False, return_target=True, num_workers = args.num_workers)

    image_encoder = ImageEncoder_MulGate_3_2Seq_SA_multiple_sentence_attention_map_soft_multiple(args.embed_size).cuda()
    caption_encoder = DummyCaptionEncoder_without_embed_multiple_random_embeddings(vocab_size=len(vocab), vocab_embed_size=2*args.embed_size,
                                              embed_size=args.embed_size).cuda()



    image_encoder.load_state_dict(torch.load("models/imagenet_randomemb_image_" + data + ".pth",map_location='cuda:0'))

    caption_encoder.load_state_dict(torch.load("models/imagenet_randomemb_text_" + data + ".pth",map_location='cuda:0'))

    image_encoder.eval()
    caption_encoder.eval()

    results = eval_batch(data_loader_dev, image_encoder, caption_encoder, ranker)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='dress')
    parser.add_argument('--embed_size', type=int , default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    evaluate(args)