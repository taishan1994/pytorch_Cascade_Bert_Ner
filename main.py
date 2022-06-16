import os
import logging
import numpy as np
import torch
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import config
import dataset_cascade
# 要显示传入BertFeature
from preprocess import BertFeature
import bert_ner_model_cascade
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
from collections import Counter, defaultdict
from tensorboardX import SummaryWriter
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score


args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)
# writer = SummaryWriter(log_dir='./logs')


def bio_decode(bio_pred, att_pred, text, id2label):
  res = defaultdict(list)
  tmp = ""
  for i, (tex, pre) in enumerate(zip(text, bio_pred)):
    if pre == 1:
      j = i+1
      while j <= len(text) - 1 and bio_pred[j] == 2:
        j += 1
      entity = text[i:j]
      entity_type_list = att_pred[i:j]
      # 计数来决定该实体属于什么类型
      entity_type_list_counter = Counter(entity_type_list).most_common(1)
      if entity_type_list_counter:
        entity_type = entity_type_list_counter[0][0]
        entity_type = id2label[entity_type]
        # 这里可以考虑是否合并相邻的同类型的实体，目前暂未这么做
        res[entity_type].append([entity, i])
      else:
        print("未知{}的类型".format(entity))
      
  return res

def convert_to_bio(entities, text):
    res = ['O'] * len(text)
    for k,values in entities.items():
      entity_type = k
      for v in values:
        res[v[1]] = 'B-{}'.format(entity_type)
        for j in range(v[1]+1, len(v[0])):
          res[j] = 'I-{}'.format(entity_type)
    return res




class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag, id2label):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        self.id2label = id2label
        model = bert_ner_model_cascade.BertNerModel(args)
        self.model, self.device = trainUtils.load_model_and_parallel(model, args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs
        self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 100 #每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)
                loss, bio_logits, att_logits = self.model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'], 
                                       batch_data['bio_labels'], batch_data['att_labels'])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                # writer.add_scalar('train/loss', loss.item(), global_step)
                global_step += 1
                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                    if f1_score > best_f1:
                        trainUtils.save_model(self.args, self.model, model_name, global_step)
                        best_f1 = f1_score
                

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            bio_pred_label = []
            att_pred_label = []
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_loss, bio_logits, att_logits = self.model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'], 
                           dev_batch_data['bio_labels'], dev_batch_data['att_labels'])
                tot_dev_loss += dev_loss.item()
                # print(bio_logits)
                # print(att_logits.shape)

                bio_batch_output = bio_logits.detach().cpu().numpy()
                bio_batch_output = np.argmax(bio_batch_output, -1)
                att_batch_output = att_logits.detach().cpu().numpy()
                att_batch_output = np.argmax(att_batch_output, -1)


                if len(bio_pred_label) == 0:
                    bio_pred_label = bio_batch_output
                    att_pred_label = att_batch_output
                else:
                    bio_pred_label = np.append(bio_pred_label, bio_batch_output, axis=0)
                    att_pred_label = np.append(att_pred_label, att_batch_output, axis=0)
            

            total_pred_entities = []
            total_gt_entities = []
            for bio_pred, att_pred, tmp_callback in zip(bio_pred_label, att_pred_label, dev_callback_info):
              text, gt_entities = tmp_callback
              tmp_metric = np.zeros([len(self.id2label), 3])  # -1是为了去掉UNK的类别
              pred_entities = dict(bio_decode(bio_pred[1:len(text)+1], att_pred[1:len(text)+1], text, self.id2label))
              total_pred_entities.append(convert_to_bio(pred_entities, text))
              total_gt_entities.append(convert_to_bio(gt_entities, text))

            precision = precision_score(total_gt_entities, total_pred_entities)
            recall = recall_score(total_gt_entities, total_pred_entities)
            f1 = f1_score(total_gt_entities, total_pred_entities)
            precision = precision_score(total_gt_entities, total_pred_entities)
            return tot_dev_loss, precision, recall, f1

    def test(self, model_path):
        model = bert_ner_model_cascade.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        bio_pred_label = []
        att_pred_label = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, bio_logits, att_logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'], 
                           dev_batch_data['bio_labels'], dev_batch_data['att_labels'])
                # print(bio_logits)
                # print(att_logits.shape)
 
                bio_batch_output = bio_logits.detach().cpu().numpy()
                bio_batch_output = np.argmax(bio_batch_output, -1)
                att_batch_output = att_logits.detach().cpu().numpy()
                att_batch_output = np.argmax(att_batch_output, -1)

                if len(bio_pred_label) == 0:
                    bio_pred_label = bio_batch_output
                    att_pred_label = att_batch_output
                else:
                    bio_pred_label = np.append(bio_pred_label, bio_batch_output, axis=0)
                    att_pred_label = np.append(att_pred_label, att_batch_output, axis=0)
            

            total_pred_entities = []
            total_gt_entities = []
            for bio_pred, att_pred, tmp_callback in zip(bio_pred_label, att_pred_label, dev_callback_info):
              text, gt_entities = tmp_callback
              tmp_metric = np.zeros([len(self.id2label), 3])  # -1是为了去掉UNK的类别
              pred_entities = dict(bio_decode(bio_pred[1:len(text)+1], att_pred[1:len(text)+1], text, self.id2label))
              total_pred_entities.append(convert_to_bio(pred_entities, text))
              total_gt_entities.append(convert_to_bio(gt_entities, text))
             
            logger.info(classification_report(total_gt_entities, total_pred_entities))

    def predict(self, raw_text, model_path):
        model = bert_ner_model_cascade.BertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
            encode_dict = tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding='max_length',
                                    truncation='longest_first',
                                    is_pretokenized=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            bio_logits, att_logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None, None)
            bio_batch_output = bio_logits.detach().cpu().numpy()
            bio_pred_label = np.argmax(bio_batch_output, -1)
            att_batch_output = att_logits.detach().cpu().numpy()
            att_pred_label = np.argmax(att_batch_output, -1)
            for bio_pred, att_pred in zip(bio_pred_label, att_pred_label):
                pred_entities = dict(bio_decode(bio_pred[1:len(raw_text)+1], att_pred[1:len(raw_text)+1], raw_text, self.id2label))
                print(pred_entities)

if __name__ == '__main__':
    data_name = 'c'
    args.train_batch_size = 32
    args.max_seq_len = 150
    model_name = ''
    if args.use_lstm == 'True' and args.use_crf == 'False':
        model_name = 'bert_bilstm'
    if args.use_lstm == 'True' and args.use_crf == 'True':
        model_name = 'bert_bilstm_crf'
    if args.use_lstm == 'False' and args.use_crf == 'True':
        model_name = 'bert_crf'
    if args.use_lstm == 'False' and args.use_crf == 'False':
        model_name = 'bert'
    commonUtils.set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))
    if data_name == "c":

        args.data_dir = './data/cner_cascade'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        print(label2id, id2label)
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset_cascade.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset_cascade.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        test_dataset = dataset_cascade.NerDataset(test_features)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        bertForNer = BertForNer(args, train_loader, dev_loader, test_loader, id2query, id2label)
        bertForNer.train()

        model_path = './checkpoints/{}/model.pt'.format(model_name)
        bertForNer.test(model_path)
        
        raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)

    if data_name == "chip":
        args.data_dir = './data/CHIP2020'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset_cascade.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset_cascade.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)

        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}/model.pt'.format(model_name)
        bertForNer.test(model_path)
        
        raw_text = "大动脉转换手术要求左心室流出道大小及肺动脉瓣的功能正常，但动力性左心室流出道梗阻并非大动脉转换术的禁忌证。"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)

    if data_name == "clue":
        args.data_dir = './data/CLUE'
        data_path = os.path.join(args.data_dir, 'final_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset_cascade.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset_cascade.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        # test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        # test_dataset = dataset.NerDataset(test_features)
        # test_loader = DataLoader(dataset=test_dataset,
        #                         batch_size=args.eval_batch_size,
        #                         num_workers=2)

        bertForNer = BertForNer(args, train_loader, dev_loader, dev_loader, id2query)
        bertForNer.train()

        model_path = './checkpoints/{}_clue/model.pt'.format(model_name)
        bertForNer.test(model_path)
        
        raw_text = "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)


