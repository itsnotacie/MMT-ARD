import torch
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group("nccl")
    dist.barrier()
    print("Passed barrier")

    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print("All reduce succeeded:", tensor.item())
    dist.destroy_process_group()

