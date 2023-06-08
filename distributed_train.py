import torch.distributed as dist
import torch.nn as nn
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PeftModel,
    PeftConfig,
    default_data_collator,
    get_linear_schedule_with_warmup,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
)

def main():
    # 超参数
    num_epochs = 10
    learning_rate =5e-5
    each_gpu_batch_size = 10
    num_epochs = 10
    accumulation_steps = 4

    backend = 'nccl'
    ''' 
    rank = 0 # 表示进程序号，用于进程间通信，可以用于表示进程的优先级
    world_size = 8 # 参与作业的进程数
    local_rank = 0 # 进程内 GPU 编号 比方说， rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU
    注意初始化rank和world_size
    你需要确保, 不同机器的rank值不同, 但是主机的rank必须为0, 而且使用init_method的ip一定是rank为0的主机, 
    其次world_size是你的进程数量, 你不能随便设置这个数值,它的值一般设置为每个节点的gpu卡个数乘以节点个数。
    '''
    
    # 初始化
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", help = "Rank of the current process in the distributed training group.",default=0, type=int)
    parser.add_argument("--world_size", help = "Total number of processes participating in the distributed training.",default=1, type=int)
    parser.add_argument("--master_addr", help = "The IP address of the master node.",default="127.0.0.1", type=str)
    parser.add_argument("--master_port", help = "The port number on the master node.",default="12355", type=str)
    args = parser.parse_args()

    dist.init_process_group(backend = backend, 
                            init_method=f'tcp://{args.master_addr}:{args.master_port}',
                                rank=args.rank, world_size=args.world_size)

    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 创建 DDP 模型
    model = ...  # Initialize your model

    # 4) 封装之前要把模型移到对应的gpu
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
    
    # 数据加载
    dataset = ...  # Initialize your dataset
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset, batch_size = args.world_size*each_gpu_batch_size, sampler=sampler)


    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=1000,
    )
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 训练循环
    for epoch in range(num_epochs):
    
        for i, (inputs, targets) in enumerate(data_loader):
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            # 使用GradScaler缩放损失，然后反向传播梯度
            scaler.scale(loss).backward()

            if i+1 % accumulation_steps == 0:  # Only update every `accumulation_steps` batches
                scaler.step(optimizer)
                scaler.update()
                # 只有在更新模型参数时，才调用学习率调度器的.step()方法
                lr_scheduler.step()
                model.zero_grad()


        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), 'model.pth')


