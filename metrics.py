import torch

class Mean:
    def __init__(self):
        self.name = "Mean"
        self.__accumulated_value = 0.0
        self.__accumulation_steps = 0.0
        self.__updated = False
        self.__average = 0.0

    def average(self):
        if self.__updated:
            self.__average = self.__accumulated_value/self.__accumulation_steps
            self.__updated = False

        return self.__average

    def update_average(self, value):
        self.__accumulated_value += torch.sum(value)
        self.__accumulation_steps += torch.numel(value)
        self.__updated = True

    def reset_states(self):
        self.__accumulated_value = 0.0
        self.__accumulation_steps = 0.0

class PSNR(Mean):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        super(PSNR, self).__init__()
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2,dim=(1,2,3))
        batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse,))
        return batch_psnr

