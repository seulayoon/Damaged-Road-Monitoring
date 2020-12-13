"""CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py"""

backbone = 'xception'
out_stride = 8  # network output stride (default: 8)
workers = 16
pretrained = True  # whether to use pretrained Xception backbone

# resize_height = 512  # model input shape
# resize_width = 1024
resize_height = 256  # model input shape
resize_width = 512

cuda = True

# If you want to use gpu:1,2,3, run CUDA_VISIBLE_DEVICES=1,2,3 python3 ...
# with gpu_ids option [0,1,2] starting with zero
# gpu_ids = [0, 1, 2, 3]  # use which gpu to train
gpu_ids = [0]

sync_bn = True if len(gpu_ids) > 1 else False  # whether to use sync bn
freeze_bn = False  # whether to freeze bn parameters (default: False)

epochs = 50
start_epoch = 0
# batch_size = 2 * len(gpu_ids)
# test_batch_size = 2 * len(gpu_ids)
batch_size = 2
test_batch_size = 2

loss_type = 'ce'  # 'ce': CrossEntropy, 'focal': Focal Loss
use_balanced_weights = False  # whether to use balanced weights (default: False)
lr = 1e-3

# Adam optimizer performed far better.
# lr_scheduler = 'poly'  # lr scheduler mode: ['poly', 'step', 'cos']
# momentum = 0.9
# weight_decay = 5e-4
# nesterov = False

resume = False  # True: load checkpoint model. False: train from scratch
checkpoint = './run/surface/deeplab/checkpoint.pth.tar'

checkname = "deeplab"  # set the checkpoint name

ft = False  # finetuning on a different dataset
eval_interval = 1  # evaluuation interval (default: 1)
no_val = False  # skip validation during training

dataset = 'surface'
root_dir = ''
if dataset == 'pascal':
    use_sbd = False  # whether to use SBD dataset
    root_dir = '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
elif dataset == 'sbd':
    root_dir = '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
elif dataset == 'cityscapes':
    root_dir = '/path/to/datasets/cityscapes/'  # foler that contains leftImg8bit/
elif dataset == 'coco':
    root_dir = '/home/piai/A2 PROJECT/coco/'
elif dataset == 'surface':
    root_dir = '/home/piai/A2 PROJECT/dataset/surface6' # 경로 설정 (dataset=='surface'만 사용)
else:
    print('Dataset {} not available.'.format(dataset))
    raise NotImplementedError


labels = [
    'background',
    'bike_lane',
    'caution_zone',
    'crosswalk',
    'guide_block',
    'roadway',
    'sidewalk',
    'damaged'
]

colors = [
    [0,0,0], #background
    [0,0,255], #bike_lane,
    [255,192,0], #caution_zone
    [255,0,255], #crosswalk
    [255,255,0], #guide_block
    [255,128,255], #roadway
    [0,255,0], #sidewalk
    [255,0,0] #damaged
]

num_classes = len(colors)  # 8
