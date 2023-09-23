import os
import logging
import sys

sys.path.append('../')
from utils import logger_init
from model import BertConfig
from model import BertForPretrainingModel
from utils import LoadBertPretrainingDataset
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== wike2 数据集相关配置
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        # self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        # self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        # self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        # self.data_name = 'wiki2'

        # ========== songci 数据集相关配置
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'SongCi')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        # self.train_file_path = os.path.join(self.dataset_dir, 'songci.train.txt')
        # self.val_file_path = os.path.join(self.dataset_dir, 'songci.valid.txt')
        # self.test_file_path = os.path.join(self.dataset_dir, 'songci.test.txt')
        # self.data_name = 'songci'

        # TODO IMPORTANT ! 需要注意 TASK NAME 的配置 """
        """ 这里的 _fzm0810_task_ 是Giikin-8.10执行训练使用的数据以及存储的模型ID；postfix参数唯一设置在训练之前数据的读取与预处理中；"""
        """ 未来推理时，该postfix标识将是加载模型的唯一凭证,设置路径：utils/create_pretraining_data.py - def load_train_val_test_data()  """
        """ 且数据预处理结束后：最终输入到Model中做MLM、NSP的原始数据集的格式为[[],[],[],...,[],[]] """
        # ========== Giikin 数据集配置
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'GiikinAds')
        self.train_file_path = os.path.join(self.dataset_dir, 'giikin.train_fzm0810_task_.pickle')
        self.val_file_path = os.path.join(self.dataset_dir, 'giikin.valid_fzm0810_task_.pickle')
        self.test_file_path = os.path.join(self.dataset_dir, 'giikin.test_fzm0810_task_.pickle')
        self.data_name = 'giikin'

        # todo train  load  base model
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")

        # todo 不同数据nameID训练的模型存储在不同的路径之下
        # self.model_save_dir = os.path.join(self.project_dir, 'cache')  # 最原始的 trained 模型存储路径
        """ 修改位置之一：训练以及：训练完毕后，做推理时加载Trained model的路径, # todo 如果需要切换数据集(不同任务)，只需要更改下面路径配置 """
        tmpsavePath = os.path.join(self.project_dir, 'cache')
        self.model_save_dir = os.path.join(tmpsavePath, '_fzm0810_task_model')
        self.pretrained_model_dir = '/home/fzm/_giikin_pro_bertmodelserver/BertWithPretrained/cache/_fzm0810_task_model'
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')

        # 其它配置
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}.bin')
        self.writer = SummaryWriter(f"runs/{self.data_name}")
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 64  # default = 16
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 100
        self.model_val_per_epoch = 1

        logger_init(log_file_name=self.data_name, log_level=self.log_level, log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def accuracy(mlm_logits, nsp_logits, mlm_labels, nsp_label, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    tmp1 = nsp_logits.argmax(1)  # isNext 判断结果
    tmp2 = (nsp_logits.argmax(1) == nsp_label)
    tmp3 = (nsp_logits.argmax(1) == nsp_label).float()
    tmp4 = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_total = len(nsp_label)
    nsp_acc = float(nsp_correct) / nsp_total
    return [mlm_acc, mlm_correct, mlm_total, nsp_acc, nsp_correct, nsp_total, tmp1]


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            mlm_logits, nsp_logits = model(input_ids=b_token_ids, attention_mask=b_mask, token_type_ids=b_segs)

            result = accuracy(mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, PAD_IDX)

            # _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
            mlm_acc, mlm_cor, mlm_tot, nsp_acc, nsp_cor, nsp_tot, isNextRes = result

            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
    model.train()
    return [float(mlm_corrects) / mlm_totals, float(nsp_corrects) / nsp_totals]


def myInferenceEvaluate(config, data_iter, model, PAD_IDX):
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals, giikin_need_isNextRes = 0, 0, 0, 0, []
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
            giikin_need_isNextRes.append([])
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            mlm_logits, nsp_logits = model(input_ids=b_token_ids, attention_mask=b_mask, token_type_ids=b_segs)
            result = accuracy(mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, PAD_IDX)
            # _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
            mlm_acc, mlm_cor, mlm_tot, nsp_acc, nsp_cor, nsp_tot, isNextRes = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
            giikin_need_isNextRes[idx].append(isNextRes.cpu().numpy().tolist())
    return [float(mlm_corrects) / mlm_totals, float(nsp_corrects) / nsp_totals, giikin_need_isNextRes]


def train(config):
    model = BertForPretrainingModel(config, config.pretrained_model_dir)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(
        vocab_path=config.vocab_path,
        tokenizer=bert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_index,
        is_sample_shuffle=config.is_sample_shuffle,
        random_state=config.random_state,
        data_name=config.data_name,
        masked_rate=config.masked_rate,
        masked_token_rate=config.masked_token_rate,
        masked_token_unchanged_rate=config.masked_token_unchanged_rate,
        # seps='，'
    )
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(
        test_file_path=config.test_file_path,
        train_file_path=config.train_file_path,
        val_file_path=config.val_file_path
    )

    # Optimizer: Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, int(len(train_iter) * 0), int(config.epochs * len(train_iter)), last_epoch=last_epoch
    )
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            loss, mlm_logits, nsp_logits = model(
                input_ids=b_token_ids,
                attention_mask=b_mask,
                token_type_ids=b_segs,
                masked_lm_labels=b_mlm_label,
                next_sentence_labels=b_nsp_label
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            # 训练阶段，返回的isNextRes无需关心，是为了可视化结果，最后测试或者使用该服务时，真正需要观察确认返回的isNextRes时再使用
            mlm_acc, _, _, nsp_acc, _, _, isNextRes = accuracy(
                mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, data_loader.PAD_IDX
            )
            if idx % 20 == 0:
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f}, nsp acc: {nsp_acc:.3f}")
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'NSP': nsp_acc, 'MLM': mlm_acc},
                                          global_step=scheduler.last_epoch)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc, nsp_acc = evaluate(config, val_iter, model, data_loader.PAD_IDX)
            logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, NSP Accuracy on val: {round(nsp_acc, 4)}")
            config.writer.add_scalars(
                main_tag='Testing/Accuracy',
                tag_scalar_dict={'NSP': nsp_acc, 'MLM': mlm_acc},
                global_step=scheduler.last_epoch
            )

            if mlm_acc > max_acc:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch, 'model_state_dict': state_dict}, config.model_save_path)


def pretty_print_mlm(token_ids, logits, pred_idx, itos, sentences, language):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
    logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
    y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
    sep = " " if language == 'en' else ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
        sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f" ### 原始: {sentence}")
        logging.info(f"  ## 掩盖: {sen_mask}")
        for idx in y_idx:
            sen[idx] = itos[y[idx]].replace("##", "")
        sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
        sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f"  ## 预测: {sen}")
        logging.info("===============")


def inference(config, sentences=None, masked=False, language='zh', random_state=None):
    """
    :param config:
    :param sentences:
    :param masked: 推理时的句子是否Mask
    :param language: 语种
    :param random_state:  控制mask字符时的随机状态
    :return:
    """
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(
        vocab_path=config.vocab_path,
        tokenizer=bert_tokenize,
        pad_index=config.pad_index,
        random_state=config.random_state,
        masked_rate=0.15  # 15% Mask掉
    )
    token_ids, pred_idx, mask = data_loader.make_inference_samples(
        sentences, masked=masked, language=language, random_state=random_state
    )
    model = BertForPretrainingModel(config, config.pretrained_model_dir)
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型进行推理......")
    else:
        raise ValueError(f"模型 {config.model_save_path} 不存在！")
    model = model.to(config.device)
    model.eval()
    with torch.no_grad():
        token_ids = token_ids.to(config.device)  # [src_len, batch_size]
        mask = mask.to(config.device)
        mlm_logits, nsp_logits = model(input_ids=token_ids, attention_mask=mask)
    pretty_print_mlm(token_ids, mlm_logits, pred_idx, data_loader.vocab.itos, sentences, language)


def myInference(config, inferenceData=''):
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize

    model = BertForPretrainingModel(config, config.pretrained_model_dir)
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行Testing......")
    model = model.to(config.device)
    model.eval()

    data_loader = LoadBertPretrainingDataset(
        vocab_path=config.vocab_path,
        tokenizer=bert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_index,
        is_sample_shuffle=config.is_sample_shuffle,
        random_state=config.random_state,
        data_name=config.data_name,
        masked_rate=config.masked_rate,
        masked_token_rate=config.masked_token_rate,
        masked_token_unchanged_rate=config.masked_token_unchanged_rate,
        # seps='，'
    )
    """
    readme:
        TTAIN PROCESS 
            pass
        TEST PROCESS 使用
            1. test_iter = data_loader.load_train_val_test_data(test_file_path=config.test_file_path, only_test=True)
        ONLINE INFERENCE 使用
            2. test_iter = data_loader.load_train_val_test_data_online(test_file_path=inferenceData, only_test=True)
    """
    """ IMPORTANT !!!!!!! 标准  TEST 时使用  """
    # todo IMPORTANT change only inference 时需要更改only_test参数；这里测试时务必需要修改   |  模型测试时使用
    # test_iter = data_loader.load_train_val_test_data(test_file_path=config.test_file_path, only_test=True)

    """ 下面仅作inference时使用 """
    # todo only online inference    |   线上做服务推理时使用； 区别在于不用再读取本地的Test文件了，是外部输入的测试数据
    test_iter = data_loader.load_train_val_test_data_online(test_file_path=inferenceData, only_test=True)

    # =====================================================================================================
    mlm_acc, nsp_acc, giikin_need_isNextResList = myInferenceEvaluate(config, test_iter, model, data_loader.PAD_IDX)
    logging.info(f" ### MLM Accuracy on test: {round(mlm_acc, 4)}, "
                 f"     NSP Accuracy on test: {round(nsp_acc, 4)}\n"
                 f"     isNextResult: {giikin_need_isNextResList}")

    # for eachBatch in giikin_need_isNextResList:
    #     print(eachBatch)


if __name__ == '__main__':
    config = ModelConfig()

    # todo only train
    # train(config)

    # wait todo only inference 需要构造测试的数据集结构；即当客户端发送了测试数据时，要首先将用户发送的数据构造成标准的数据格式，才能送入模型
    # 测试线上推理的假设数据
    inferenceData = [
        ['彩色的条纹与淡雅的裙子组合在一起十分和谐', '充满清新可爱的感觉'],
        # ['精致做工走线整齐', '剪裁线条流畅', '细节之处见品质'],
        ['精致做工走线整齐', '剪裁线条流畅'],
        # ['而破洞元素则有着街头风的随性不羁', '张扬出独特潮流个性', '裤脚运用毛边设计', '更突显出随性休闲的潮流韵味'],
        ['而破洞元素则有着街头风的随性不羁', '张扬出独特潮流个性'], ['裤脚运用毛边设计', '更突显出随性休闲的潮流韵味'],
        # ['横条立体装饰更凸显完美曲线', '包臀弹力面料舒适柔软', '领部立体饰品装饰', '彰显优雅气质'],
        ['横条立体装饰更凸显完美曲线', '包臀弹力面料舒适柔软'], ['领部立体饰品装饰', '彰显优雅气质'],
        # ['别致的荷叶边造型', '更突显出甜美浪漫的少女情怀', '带来眼前一亮的减龄效果']
        ['别致的荷叶边造型', '更突显出甜美浪漫的少女情怀']
    ]
    myInference(config, inferenceData)

    # todo inference  模型config部分和模型加载部分有更改
    sentences_1 = ["I no longer love her, true, but perhaps I love her.", "Love is so short and oblivion so long."]
    sentences_2 = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。", "红酥手。黄藤酒。满园春色宫墙柳。"]
    sentences_3 = ["恢复肌肤自然水感", "舒缓肌肤表皮的干燥感"]
    sentences_4 = ["舒缓肌肤表皮的干燥感"]
    inference(config, sentences_3, masked=False, language='zh', random_state=2022)
