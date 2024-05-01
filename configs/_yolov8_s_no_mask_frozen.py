_base_ = '/disk/xxiong52/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`
# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/disk/xxiong52/Dataset/wildfire/smoke/smoke_img/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images/'  # Prefix of val image path
# Path of test annotation file
test_ann_file = 'annotations/test.json'
test_data_prefix = 'images/'  # Prefix of test image path

#dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset.
class_name = ('smoke', ) # Class names
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

num_classes = len(class_name)  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.001
max_epochs = 100  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

num_epochs_stage2 = 5

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100-5881dec4.pth"

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))
            
# train_pipeline = [
#     *pre_transform, *mosaic_affine_transform,
#     dict(
#         type='YOLOv5MixUp',
#         prob=mixup_prob,
#         pre_transform=[*pre_transform, *mosaic_affine_transform]),
#     *last_transform
# ]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix))
    )

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

val_evaluator = dict(  # Validation evaluator config
    ann_file=data_root + val_ann_file,  # Annotation file path
    )

test_dataloader = val_dataloader
test_evaluator = val_evaluator  # Testing evaluator config
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# param_scheduler =[
#     dict(
#         type = 'LinearLR',
#         start_factor=1e-5,
#         by_epoch=False,
#         begin = 0,
#         end = 20),
#     dict(
#         type = 'CosineAnnealingLR',
#         eta_min=base_lr * 0.05,
#         begin = max_epochs//2,
#         end=max_epochs,
#         T_max=max_epochs//2,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]

# optim_wrapper = dict(optimizer=dict(lr=base_lr))

# _base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2
# _base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# default_hooks = dict(
#     # Save weights every 10 epochs and a maximum of two weights can be saved.
#     # The best model is saved automatically during model evaluation
#     checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
#     # The warmup_mim_iter parameter is critical.
#     # The default value is 1000 which is not suitable for cat datasets.
#     param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
#     # The log printing interval is 5
#     logger=dict(type='LoggerHook', interval=5))
# # The evaluation interval is 10
# train_cfg = dict(max_epochs=max_epochs, val_interval=10)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])