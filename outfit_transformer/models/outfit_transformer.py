# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Rohan Sarkar, Navaneeth Bodla, et al. Outfit Transformer : Outfit Representations for Fashion Recommendation. CVPR, 2023. 
    (https://arxiv.org/abs/2204.04812)
"""
import os
import math
import wandb
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from outfit_transformer.utils.utils import *
from outfit_transformer.models.encoder.builder import *
from outfit_transformer.loss.focal_loss import focal_loss
from outfit_transformer.loss.triplet_loss import outfit_triplet_loss, triplet_loss


class OutfitTransformer(nn.Module):
    
    def __init__(
            self,
            embedding_dim: int = 128,
            img_backbone: str = 'resnet-18',
            txt_backbone: Optional[str] = 'bert',
            txt_huggingface: str = 'sentence-transformers/paraphrase-albert-small-v2',
            nhead = 16,
            dim_feedforward = 512, # Not specified; use 4x the built-in size the same as BERT
            num_layers = 6,
            freeze_image_backbone=False
            ):
        super().__init__()
        if txt_backbone:
            self.encode_dim = embedding_dim // 2
        else:
            self.encode_dim = embedding_dim
        self.embedding_dim = embedding_dim
        #------------------------------------------------------------------------------------------------#
        # Encoder
        self.img_encoder = build_img_encoder(
            backbone = img_backbone, 
            embedding_dim = self.encode_dim,
            do_linear_probing=freeze_image_backbone
            )
        if txt_backbone:
            self.txt_encoder = build_txt_encoder(
                backbone = txt_backbone, 
                embedding_dim = self.encode_dim, 
                huggingface = txt_huggingface, 
                do_linear_probing = True
                )
        else:
            self.txt_encoder = None

        #------------------------------------------------------------------------------------------------#
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            batch_first = True
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = encoder_layer, 
            num_layers = num_layers, 
            norm = nn.LayerNorm(self.embedding_dim)
            )
        #------------------------------------------------------------------------------------------------#
        # Embeddings
        # For CP
        self.cp_embedding = nn.Parameter(torch.empty((1, self.embedding_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cp_embedding.data)
        # For CIR
        # This part is different from the original paper. I replaced it with token instead of image.
        self.cir_embedding = nn.Parameter(torch.empty((1, self.encode_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cir_embedding.data)
        #------------------------------------------------------------------------------------------------#
        # FC layers for classification(for pre-training)
        self.fc_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
            )
        #------------------------------------------------------------------------------------------------#
        # FC layers for projection(for fine-tuning)
        self.fc_projection = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
            )
        #------------------------------------------------------------------------------------------------#
        self.logs = {
            'epochs': defaultdict(lambda: {'train_loss': [], 'valid_loss': []})
        }


    def encode(self, inputs, no_unstack=False):
        inputs = stack_dict(inputs)
        img_embedddings = [self.img_encoder(inputs['image_features'])]
        if self.txt_encoder is not None:
            txt_embedddings = [self.txt_encoder(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask']
                )]
        else:
            txt_embedddings = []
        general_embeddings = torch.concat(img_embedddings + txt_embedddings, dim=-1)
        outputs = {
            'mask': inputs['mask'],
            'embed': general_embeddings
            }
        if no_unstack:
            return outputs
        else:
            return unstack_dict(outputs)


    def cp_forward(self, inputs, do_encode=False):
        if do_encode:
            x = self.encode(inputs)
        else:
            x = inputs

        embed = torch.cat([
            self.cp_embedding.unsqueeze(0).expand(len(x['embed']), -1, -1),
            x['embed']
            ], dim=1)
        src_key_padding_mask = torch.cat([
            torch.zeros((len(x['mask']), 1), device=embed.device), 
            x['mask']
            ], dim=1).bool()
        y = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)[:, 0, :]
        y = self.fc_classifier(y)
        logits = F.sigmoid(y)
        return logits

    def cir_forward(self, batch, device):
        batch_size = batch['outfits']['mask'].shape[0]

        n_outfits = batch['outfits']['mask'].shape[1]
        n_positive = batch['positive']['mask'].shape[1]
        n_negatives = batch['negatives']['mask'].shape[1]

        concatenated_inputs = {
            key: torch.cat((
                batch['outfits'][key],
                batch['positive'][key],
                batch['negatives'][key]
            ), dim=1)
            for key in batch['outfits'].keys()
        }

        device_inputs = {key: value.to(device) for key, value in concatenated_inputs.items()}
        
        encoded_inputs = self.encode(device_inputs)

        encoded_outfit, encoded_positive, encoded_negatives = (
            {
                k: encoded_inputs[k].split([n_outfits, n_positive, n_negatives], dim=1)[i]
                for k in encoded_inputs
            }
            for i in (0, 1, 2)
        )

        # The query is always at the front, see PolyvoreDatasetCir
        y = self.transformer(
            encoded_outfit['embed'],
            src_key_padding_mask=encoded_outfit['mask'].bool()
        )[range(batch_size), 0, :]
        y = self.fc_projection(y)

        return y, encoded_positive['embed'], encoded_negatives['embed']

    def forward(self, inputs):
        return self.cp_forward(inputs, do_encode=True)
 

    def iteration_step(self, batch, task, device):
        if task == 'cp':
            targets = batch['targets'].to(device)
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}

            logits = self.cp_forward(inputs, do_encode=True)
            loss = focal_loss(logits, targets.to(device))
        elif task == 'cir':
            y, positive_embed, negative_embeds = self.cir_forward(batch, device)
            loss = outfit_triplet_loss(y, positive_embed, negative_embeds, margin=2)
            
        return loss


    def iteration(self, dataloader, epoch, is_train, device,
                  optimizer=None, scheduler=None, use_wandb=False, task='cp'):
        type_str = f'{task} train' if is_train else f'{task} valid'
        epoch_iterator = tqdm(dataloader)
        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            #--------------------------------------------------------------------------------------------#
            # Compute Loss
            running_loss = self.iteration_step(batch, task, device)
            if is_train:
                self.logs['epochs'][epoch]['train_loss'].append(running_loss)
            else:
                self.logs['epochs'][epoch]['valid_loss'].append(running_loss)
            #--------------------------------------------------------------------------------------------#
            # Backward
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            #--------------------------------------------------------------------------------------------#
            # Logging
            total_loss += running_loss.item()
            epoch_iterator.set_description(
                f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_loss': running_loss, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                if is_train == True:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_loss = total_loss / iter
        print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')
        return total_loss
    
    
    def fit(
            self,
            task: Literal['cp', 'cir'],
            save_dir: str,
            n_epochs: int,
            optimizer: torch.optim.Optimizer,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            valid_cp_dataloader: Optional[DataLoader] = None,
            valid_fitb_dataloader: Optional[DataLoader] = None,
            valid_cir_dataloader: Optional[DataLoader] = None,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,
            device: Literal['cuda', 'cpu'] = 'cuda',
            use_wandb: Optional[bool] = False,
            save_every: Optional[int] = 1
            ):
        
        date = datetime.now().strftime('%Y-%m-%d')
        save_dir = os.path.join(save_dir, date)

        device = torch.device(0) if device == 'cuda' else torch.device('cpu')
        self.to(device)

        best_criterion = -np.inf
        best_model = None

        for epoch in range(n_epochs):
            #--------------------------------------------------------------------------------------------#
            self.train()
            train_loss = self.iteration(
                dataloader = train_dataloader, 
                epoch = epoch, 
                is_train = True, 
                device = device,
                optimizer = optimizer, 
                scheduler = scheduler, 
                use_wandb = use_wandb,
                task = task
                )
            #--------------------------------------------------------------------------------------------#
            self.eval()
            with torch.no_grad():
                valid_loss = self.iteration(
                    dataloader = valid_dataloader, 
                    epoch = epoch, 
                    is_train = False,
                    device = device,
                    use_wandb = use_wandb,
                    task = task
                    )
                #----------------------------------------------------------------------------------------#
                if task == 'cp':
                    cp_score = self.cp_evaluation(
                        dataloader = valid_cp_dataloader,
                        epoch = epoch,
                        is_test = False,
                        device = device,
                        use_wandb = use_wandb
                        )
                    fitb_score = self.fitb_evaluation(
                        dataloader = valid_fitb_dataloader,
                        epoch = epoch,
                        is_test = False,
                        device = device,
                        use_wandb = use_wandb
                        )
                    if cp_score > best_criterion:
                        best_criterion = cp_score
                        best_state = deepcopy(self.state_dict())
                elif task == 'cir':
                    cir_score = self.cir_evaluation(
                        dataloader = valid_cir_dataloader,
                        epoch = epoch,
                        is_test = False,
                        device = device,
                        use_wandb = use_wandb
                        )
                    if cir_score > best_criterion:
                        best_criterion = cir_score
                        best_state = deepcopy(self.state_dict())

            if epoch % save_every == 0:
                model_name = f'{epoch}_{best_criterion:.3f}'
                self._save(save_dir, model_name)
        self._save(save_dir, 'final', best_state)


    def _save(self, 
              dir, 
              model_name, 
              best_state=None):
        
        def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)

        _create_folder(dir)
        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {'state_dict': best_state if best_state is not None else self.state_dict()}
        torch.save(checkpoint, path)
        print(f'[COMPLETE] Save at {path}')


    def cp_evaluation(self, dataloader, epoch, is_test, device, use_wandb=False):
        type_str = 'cp_test' if is_test else 'cp_eval'
        epoch_iterator = tqdm(dataloader)
        
        total_targets, total_score, total_pred = [], [], []
        for iter, batch in enumerate(epoch_iterator, start=1):
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
            logits = self.cp_forward(inputs, do_encode=True)

            targets = batch['targets'].view(-1).detach().numpy().astype(int)
            run_score = logits.view(-1).cpu().detach().numpy()
            run_pred = (run_score >= 0.5).astype(int)

            total_targets = np.concatenate([total_targets, targets], axis=None)
            total_score = np.concatenate([total_score, run_score], axis=None)
            total_pred = np.concatenate([total_pred, run_pred], axis=None)
            #--------------------------------------------------------------------------------------------#
            # Logging
            run_acc = sum(targets == run_pred) / len(targets)
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Acc: {run_acc:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_acc': run_acc, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_acc = sum(total_targets == total_pred) / len(total_targets)
        total_auc = roc_auc_score(total_targets, total_score)
        print(f'[{type_str} END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f} | AUC: {total_auc:.5f}\n')
        return total_auc
    

    def fitb_evaluation(self, dataloader, epoch, is_test, device, use_wandb=False):
        type_str = 'fitb_test' if is_test else 'fitb_eval'
        epoch_iterator = tqdm(dataloader)
        
        total_pred = []
        for iter, batch in enumerate(epoch_iterator, start=1):
            questions = {key: value.to(device) for key, value in batch['questions'].items()}
            candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

            n_batch, n_candidate = candidates['mask'].shape

            encoded_question = self.encode(questions)
            encoded_candiates = self.encode(candidates)

            encoded = {
                'mask': torch.cat([
                    encoded_question['mask'].unsqueeze(1).expand(-1, n_candidate, -1), 
                    encoded_candiates['mask'].unsqueeze(2)
                    ], dim=2).flatten(0, 1),
                'embed': torch.cat([
                    encoded_question['embed'].unsqueeze(1).expand(-1, n_candidate, -1, -1), 
                    encoded_candiates['embed'].unsqueeze(2)
                    ], dim=2).flatten(0, 1)
                }

            logits = self.cp_forward(encoded)
            logits = logits.view(n_batch, n_candidate)
            run_pred = logits.argmax(dim=1).cpu().detach().numpy()
            total_pred = np.concatenate([total_pred, run_pred], axis=None)
            #--------------------------------------------------------------------------------------------#
            # Logging
            run_acc = sum(run_pred == 0) / len(run_pred)
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Acc: {run_acc:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_acc': run_acc, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_acc = sum(total_pred == 0) / len(total_pred)
        print(f'[{type_str} END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f}\n')
        return total_acc

    def build_embeddings_db(self, dataloader, device):
        # Build the embeddings database
        print("Build embeddings database")
        all_embeddings = []
        for i, batch in enumerate(tqdm(dataloader)):
            data = {
                key: value.to(device) 
                for key, value in batch['outfits'].items()
            }
            product_ids = data['item_ids'][~data['mask']]
            batch_size = data['mask'].shape[0]
            batch_index = list(range(batch_size))

            with torch.no_grad():
                embeddings = self.encode(data, no_unstack=True)
                all_embeddings.extend(
                    zip(
                        product_ids.detach().cpu(), 
                        embeddings['embed'].detach().cpu()
                    )
                )

        return {
            'embeddings': torch.stack([torch.tensor(it[1]) for it in all_embeddings]),
            'product_ids': torch.stack([it[0] for it in all_embeddings])
        }

    def cir_evaluation(self, dataloader, epoch, is_test, device, use_wandb=True):
        type_str = 'cir_test' if is_test else 'cir_eval'
        embeddings_db = self.build_embeddings_db(dataloader, device)
        epoch_iterator = tqdm(dataloader)
        overall_r_at_10 = 0
        overall_r_at_30 = 0
        overall_r_at_50 = 0

        for iter, batch in enumerate(epoch_iterator, start=1):
            with torch.no_grad():
                target_item_ids = batch['positive']['item_ids']
                batch_size = target_item_ids.shape[0]

                query_embeddings, _, _ = self.cir_forward(batch, device)

                # Calculate the distance between query and db embeddings
                distances = torch.cdist(
                    query_embeddings, embeddings_db['embeddings'].to(device)
                ).cpu()

                # Top 10, top 30, top 50
                top_10_item_ids = embeddings_db['product_ids'][distances.topk(10, dim=1).indices]
                top_30_item_ids = embeddings_db['product_ids'][distances.topk(30, dim=1).indices]
                top_50_item_ids = embeddings_db['product_ids'][distances.topk(50, dim=1).indices]

                r_at_10 = (sum(torch.isin(target_item_ids, top_10_item_ids))/batch_size).item()
                r_at_30 = (sum(torch.isin(target_item_ids, top_30_item_ids))/batch_size).item()
                r_at_50 = (sum(torch.isin(target_item_ids, top_50_item_ids))/batch_size).item()

                overall_r_at_10 = (overall_r_at_10 * (iter - 1) + r_at_10) / iter
                overall_r_at_30 = (overall_r_at_30 * (iter - 1) + r_at_30) / iter
                overall_r_at_50 = (overall_r_at_50 * (iter - 1) + r_at_50) / iter

            epoch_iterator.set_description(
                f'[{type_str}] Epoch: {epoch + 1:03} | r@10: {r_at_10:.3f}, r@30: {r_at_30:.3f}, r@50: {r_at_50:.3f}'
            )

        print(f'[{type_str} END] Epoch: {epoch + 1:03} | r@10: {overall_r_at_10:.3f}, r@30: {overall_r_at_30:.3f}, r@50: {overall_r_at_50:.3f}\n')

        return overall_r_at_10
