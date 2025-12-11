import torch
from data.pose_dataset import PoseDataset
from data.scenes_dataset import ScenesDataset

def CreateDataLoader(opt):
    if "cambridge" in opt.dataroot:
        dataset = PoseDataset(opt)
    elif "7scenes" in opt.dataroot:
        dataset = ScenesDataset(opt)
    print("dataset [%s] was created" % (type(dataset).__name__))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))
    return dataloader
