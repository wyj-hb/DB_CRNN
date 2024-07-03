import os
import gc
import time
import random
import warnings
import cv2
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.losses import DBLoss
from src.lr_schedulers import WarmupPolyLR
from models.DB_CRNN import DB_CRNN
from src.text_metrics import (cal_text_score, RunningScore, QuadMetric)
from src.utils import (dict_to_device,
                       visualize_tfb, strLabelConverter, crop_and_resize, setup_logger)
from src.postprocess import SegDetectorRepresenter
from torch.autograd import Variable
warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
convert = None
from torch.utils.data import SubsetRandomSampler

def custom_collate_fn(batch):
    image_paths = [item['image_path'] for item in batch]
    imgs = [torch.from_numpy(item['img']) for item in batch]
    prob_maps = [torch.from_numpy(item['prob_map']) for item in batch]
    masks = [torch.from_numpy(item['supervision_mask']) for item in batch]
    thresh_maps = [torch.from_numpy(item['thresh_map']) for item in batch]
    text_area_maps = [torch.from_numpy(item['text_area_map']) for item in batch]
    image_ori= [torch.from_numpy(item['img_origin']) for item in batch]
    messages = [item['message'] for item in batch]
    str_data = [item['str_data'] for item in batch]
    # 根据需要，使用 torch.stack 或其他方法处理某些字段
    imgs = torch.stack(imgs)
    prob_maps = torch.stack(prob_maps)
    masks = torch.stack(masks)
    thresh_maps = torch.stack(thresh_maps)
    text_area_maps = torch.stack(text_area_maps)
    gt_box = [msg['gt'] for msg in messages]
    texts = [msg['text'] for msg in messages]
    lengths = [msg['length'] for msg in messages]
    image_origin = [data for data in image_ori]
    str_data = [data for data in str_data]
    if 'anns' in batch[0].keys() and 'ignore_tags' in batch[0].keys():
        anns = [item['anns'] for item in batch]
        ignore_tags = [item['ignore_tags'] for item in batch]
        anns = [data for data in anns]
        ignore_tags = [data for data in ignore_tags]
        return {
            'image_path': image_paths,
            'img': imgs,
            'prob_map': prob_maps,
            'supervision_mask': masks,
            'thresh_map': thresh_maps,
            'text_area_map': text_area_maps,
            'texts': texts,
            'lengths': lengths,
            'gt': gt_box,
            'image_origin': image_origin,
            'str_data': str_data,
            'anns':anns,
            'ignore_tags':ignore_tags
        }
    else:
        return {
            'image_path': image_paths,
            'img': imgs,
            'prob_map': prob_maps,
            'supervision_mask': masks,
            'thresh_map': thresh_maps,
            'text_area_map': text_area_maps,
            'texts': texts,
            'lengths': lengths,
            'gt':gt_box,
            'image_origin':image_origin,
            'str_data':str_data
        }
def get_data_loaders(cfg):
    global convert
    convert = strLabelConverter(cfg.alphabet)
    dataset_name = cfg.dataset.name
    ignore_tags = cfg.data[dataset_name].ignore_tags
    train_dir = cfg.data[dataset_name].train_dir
    test_dir = cfg.data[dataset_name].test_dir
    train_gt_dir = cfg.data[dataset_name].train_gt_dir
    test_gt_dir = cfg.data[dataset_name].test_gt_dir
    if dataset_name == 'totaltext':
        from src.data_loaders import TotalTextDatasetIter
        TextDatasetIter = TotalTextDatasetIter

    train_iter = TextDatasetIter(train_dir,
                                 train_gt_dir,
                                 ignore_tags,
                                 convert = convert,
                                 image_size=cfg.hps.img_size,
                                 is_training=True,
                                 debug=False)
    test_iter = TextDatasetIter(test_dir,
                                test_gt_dir,
                                ignore_tags,
                                convert,
                                image_size=cfg.hps.img_size,
                                is_training=False,
                                debug=False)
    indices = random.sample(range(299), 100)
    # 使用 SubsetRandomSampler 来选择特定的样本
    sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(dataset=train_iter,
                              batch_size=cfg.hps.batch_size,
                              shuffle=True,
                              collate_fn=custom_collate_fn,
                              num_workers=1
                              )
    test_loader = DataLoader(dataset=test_iter,
                             batch_size=cfg.hps.test_batch_size,
                             collate_fn=custom_collate_fn,
                             shuffle=False,
                             sampler=sampler,

    num_workers=0)
    return train_loader, test_loader
def main(cfg):
    # TODO 配置日志
    logger = setup_logger(
        os.path.join(cfg.meta.root_dir, cfg.logging.logger_file))
    log_dir_path = os.path.join(cfg.meta.root_dir, "logs")
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    tfb_log_dir = os.path.join(log_dir_path, str(int(time.time())))
    logger.info(tfb_log_dir)
    if not os.path.exists(tfb_log_dir):
        os.makedirs(tfb_log_dir)
    # TODO 配置日志目录
    tfb_writer = SummaryWriter("/root/tf-logs")
    device = cfg.meta.device
    logger.info(device)
    # TODO 加载DB_CRNN模型
    db_rcnn = DB_CRNN(cfg).to(device)
    lr_optim = cfg.optimizer.lr
    db_rcnn.train()
    for name, param in db_rcnn.named_parameters():
        if param.requires_grad:
            print(f"{name} has requires_grad=True")
    # TODO 设置优化器
    db_optimizer = torch_optim.Adam(db_rcnn.parameters(),
                                    lr=lr_optim,
                                    weight_decay=cfg.optimizer.weight_decay,
                                    amsgrad=cfg.optimizer.amsgrad)
    # setup model checkpoint
    best_test_loss = np.inf
    best_train_loss = np.inf
    best_hmean = 0
    db_scheduler = None
    lrs_mode = cfg.lrs.mode
    logger.info("Learning rate scheduler: {}".format(lrs_mode))
    if lrs_mode == 'poly':
        db_scheduler = WarmupPolyLR(db_optimizer,
                                    warmup_iters=cfg.lrs.warmup_iters)
    elif lrs_mode == 'reduce':
        db_scheduler = torch_optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=db_optimizer,
            mode='min',
            factor=cfg.lrs.factor,
            patience=cfg.lrs.patience,
            verbose=True)
    # TODO 获取数据迭代器
    dataset_name = cfg.dataset.name
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Ignore tags: {}".format(cfg.data[dataset_name].ignore_tags))
    totaltext_train_loader, totaltext_test_loader = get_data_loaders(cfg)
    # TODO 训练模型
    logger.info("Start training!")
    torch.cuda.empty_cache()
    gc.collect()
    global_steps = 0
    for epoch in range(cfg.hps.no_epochs):
        # TRAINING
        db_rcnn.train()
        train_loss = 0
        running_metric_text = RunningScore(cfg.hps.no_classes)
        for batch_index, batch in enumerate(totaltext_train_loader):
            batch = dict_to_device(batch, device=device)
            global_steps += 1
            loss = db_rcnn(batch,istrian=True)
            prob_loss = loss['prob_loss']
            threshold_loss = loss['threshold_loss']
            binary_loss = loss['binary_loss']
            prob_threshold_loss = loss['prob_threshold_loss']
            total_loss = loss['total_loss']
            preds = loss['preds']
            # TODO 设置学习率
            lr = db_optimizer.param_groups[0]['lr']
            # TODO 梯度清零
            db_optimizer.zero_grad()
            # TODO 反向传播
            total_loss.backward()
            # TODO 更新
            db_optimizer.step()
            if lrs_mode == 'poly':
                db_scheduler.step()
            score_shrink_map = cal_text_score(
                preds[:, 0, :, :],
                batch['prob_map'],
                batch['supervision_mask'],
                running_metric_text,
                thresh=cfg.metric.thred_text_score)
            train_loss += total_loss
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']
            # TODO 记录日志信息
            # TODO 总体损失
            tfb_writer.add_scalar('TRAIN/LOSS/total_loss', total_loss,
                                  global_steps)
            # TODO 阈值损失
            tfb_writer.add_scalar('TRAIN/LOSS/loss', prob_threshold_loss,
                                  global_steps)
            # TODO 概率损失
            tfb_writer.add_scalar('TRAIN/LOSS/prob_loss', prob_loss,
                                  global_steps)
            # TODO 阈值损失
            tfb_writer.add_scalar('TRAIN/LOSS/threshold_loss', threshold_loss,
                                  global_steps)
            # TODO 二值损失
            tfb_writer.add_scalar('TRAIN/LOSS/binary_loss', binary_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/acc', acc, global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map',
                                  iou_shrink_map, global_steps)
            # TODO 学习率变化
            tfb_writer.add_scalar('TRAIN/HPs/lr', lr, global_steps)
            if global_steps % cfg.hps.log_iter == 0:
                logger.info(
                    "[{}-{}] - lr: {} - total_loss: {} - loss: {} - acc: {} - iou: {}"  # noqa
                        .format(epoch + 1, global_steps, lr, total_loss,
                                prob_threshold_loss, acc, iou_shrink_map))
                # TODO 测试
                end_epoch_loss = train_loss / len(totaltext_train_loader)
                logger.info("Train loss: {}".format(end_epoch_loss))
                gc.collect()
                # TFB IMGs
                # shuffle = True
                # TODO 训练图像展示
                visualize_tfb(tfb_writer,
                              batch['img'],
                              preds,
                              global_steps=global_steps,
                              thresh=cfg.metric.thred_text_score,
                              mode="TRAIN")
                seg_obj = SegDetectorRepresenter(thresh=cfg.metric.thred_text_score,
                                                 box_thresh=cfg.metric.prob_threshold,
                                                 unclip_ratio=cfg.metric.unclip_ratio)
                # EVAL
                # TODO 转为评估模式
                metric_cls = QuadMetric()
                db_rcnn.eval()
                test_running_metric_text = RunningScore(cfg.hps.no_classes)
                tests_loss = 0
                raw_metrics = []
                n_correct = 0
                # TODO 随机选择一个进行可视化
                test_visualize_index = random.choice(range(len(totaltext_test_loader)))
                total_predict_count = 0
                for test_batch_index, test_batch in tqdm(
                        enumerate(totaltext_test_loader),
                        total=len(totaltext_test_loader)):
                    # TODO 去除test_bacth
                    with torch.no_grad():
                        test_batch = dict_to_device(test_batch, device)
                        # TODO 测试模式,所以为False
                        test_loss = db_rcnn(test_batch, istrian=False)
                        test_preds = test_loss['preds']
                        assert test_preds.size(1) == 2
                        test_total_loss = test_loss['total_loss']
                        _batch = torch.stack([
                            test_batch['prob_map'], test_batch['supervision_mask'],
                            test_batch['thresh_map'], test_batch['text_area_map']
                        ])
                        # TODO 计算训练损失
                        tests_loss += test_total_loss
                        # TODO 预测
                        preds_test = test_loss['preds_dec']
                        # TODO 取数据
                        cpu_texts = [word for sublist in test_batch['str_data'] for word in sublist]
                        batch_size = preds_test.shape[1]
                        total_predict_count += batch_size
                        preds_size = Variable(torch.IntTensor([preds_test.size(0)] * batch_size))
                        _, preds = preds_test.max(2)
                        preds = preds.transpose(1, 0).contiguous().view(-1)
                        # TODO 解码
                        sim_preds = convert.decode(preds.data, preds_size.data, raw=False)
                        # 初始化一个空的布尔数组
                        correct_preds = [False] * len(cpu_texts)
                        # 对比预测值和目标值，记录相等的位置
                        for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts)):
                            logger.info("{}------>{}".format(pred, target.lower()))
                            # TODO 正确并且不被忽略
                            if pred == target.lower():  # 这里假设要比较的是小写形式
                                correct_preds[i] = True
                                n_correct + 1
                        # TODO 随机可视化
                        if test_batch_index == test_visualize_index:
                            visualize_tfb(tfb_writer,
                                          test_batch['img'],
                                          test_preds,
                                          global_steps=global_steps,
                                          thresh=cfg.metric.thred_text_score,
                                          mode="TEST")
                        # TODO
                        test_score_shrink_map = cal_text_score(
                            test_preds[:, 0, :, :],
                            test_batch['prob_map'],
                            test_batch['supervision_mask'],
                            test_running_metric_text,
                            thresh=cfg.metric.thred_text_score)
                        test_acc = test_score_shrink_map['Mean Acc']
                        tfb_writer.add_scalar('TEST/LOSS/val_loss', test_total_loss,
                                              global_steps)
                        tfb_writer.add_scalar('TEST/ACC_IOU/val_acc', test_acc,
                                              global_steps)
                        # TODO 计算 P/R/Hmean
                        batch_shape = [(cfg.hps.img_size, cfg.hps.img_size) for _ in range(cfg.hps.test_batch_size)]
                        batch_shape = {'shape': batch_shape}
                        box_list, score_list = seg_obj(
                            batch_shape,
                            test_preds,
                            is_output_polygon=cfg.metric.is_output_polygon)
                        raw_metric = metric_cls.validate_measure(
                            test_batch, (box_list, score_list), data=correct_preds)
                        raw_metrics.append(raw_metric)
                        metrics = metric_cls.gather_measure(raw_metrics)
                metrics = metric_cls.gather_measure(raw_metrics)
                # TODO 计算召回率、精确率和F1
                recall = metrics['recall'].avg
                precision = metrics['precision'].avg
                hmean = metrics['fmeasure'].avg
                E2E_precision = metrics['E2E_precision'].avg
                E2E_hmean = metrics['E2E_fmeasure'].avg
                E2E_recall = metrics['E2E_recall'].avg
                # raw_preds = convert.decode(preds.data, preds_size.data, raw=True)[:cfg.n_test_disp]
                # for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
                #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
                # accuracy = n_correct / float(len(totaltext_test_loader) * total_predict_count)
                # print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
                logger.info(
                    "[{}-{}] - lr: {} - total_loss: {} - loss: {} - acc: {} - iou: {}"  # noqa
                        .format(epoch + 1, global_steps, lr, total_loss,
                                prob_threshold_loss, acc, iou_shrink_map))
                # if hmean >= best_hmean:
                #     best_hmean = hmean
                #     torch.save(
                #         db_rcnn.state_dict(),
                #         os.path.join(cfg.meta.root_dir, cfg.model.best_hmean_cp_path))

                logger.info(
                    "TEST/Recall: {} - TEST/Precision: {} - TEST/HMean: {} - TEST/E2E_Precision: {} - TEST/E2E_hmean: {} - TEST/E2E_recall: {}".format(
                        recall, precision, hmean,E2E_precision,E2E_hmean,E2E_recall))
                tfb_writer.add_scalar('TEST/recall', recall, global_steps)
                tfb_writer.add_scalar('TEST/precision', precision, global_steps)
                tfb_writer.add_scalar('TEST/hmean', hmean, global_steps)
                tfb_writer.add_scalar('TEST/E2E_recall', E2E_recall, global_steps)
                tfb_writer.add_scalar('TEST/E2E_precision', E2E_precision, global_steps)
                tfb_writer.add_scalar('TEST/E2E_hmean', E2E_hmean, global_steps)
                tsloss = tests_loss / len(totaltext_test_loader)
                tfb_writer.add_scalar('TEST/test_loss', tsloss, global_steps)
                logger.info("[{}] - test_loss: {}".format(global_steps, tsloss))

                # if test_loss <= best_test_loss and train_loss <= best_train_loss:
                #     best_test_loss = test_loss
                #     best_train_loss = train_loss
                #     torch.save(DB_CRNN.state_dict(),
                #                os.path.join(cfg.meta.root_dir, cfg.model.best_cp_path))

                if lrs_mode == 'reduce':
                    db_scheduler.step(tsloss)
                torch.cuda.empty_cache()
                gc.collect()
                db_rcnn.train()
    logger.info("Training completed")
    torch.save(DB_CRNN.state_dict(),
               os.path.join(cfg.meta.root_dir, cfg.model.last_cp_path))
    logger.info("Saved model")
@hydra.main(config_path="myconfig.yaml")
def run(cfg):
    main(cfg)
if __name__ == '__main__':
    run()
