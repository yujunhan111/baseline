import torch


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # 修改为存储从加载器中获取的下一批数据的字典
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            # 获取下一批数据
            self.next_batch = next(self.loader)
            with torch.cuda.stream(self.stream):
                # 将 demographic 数据移到 GPU 上
                self.next_batch['demographic'] = self.next_batch['demographic'].cuda(non_blocking=True)
        except StopIteration:
            self.next_batch = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch