import torch
from data.pose_dataset import PoseDataset

def CreateDataLoader(opt):
    dataset = PoseDataset(opt)
    print("dataset [%s] was created" % (type(dataset).__name__))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))
    return dataloader
