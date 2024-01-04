import torch
from torchvision.transforms import functional_tensor as F


class PhotometricDistort(torch.nn.Module):
    def __init__(self):
        super(PhotometricDistort, self).__init__()
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.distort_forward = torch.nn.Sequential(*self.pd[1:])
        self.distort_backward = torch.nn.Sequential(*self.pd[:-1])

        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomChannelSwap()

    def forward(self, sample):
        if torch.rand(1).item() < 0.5:
            sample = self.rand_light_noise(self.distort_forward(self.rand_brightness(sample)))
        else:
            sample = self.rand_light_noise(self.distort_backward(self.rand_brightness(sample)))
        return sample


class RandomContrast(torch.nn.Module):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        super(RandomContrast, self).__init__()
        assert lower <= upper, f"contrast upper({upper:.2f}) must be larger than lower({lower:.2f})."
        assert 0 <= lower, f"contrast lower({lower:.2f}) must be non-negative."
        assert 0 <= p <= 1, f"probability({p:.2f}) must be in [0, 1]."

        self.lower = lower
        self.upper = upper
        self.p = p

    # expects float image
    def forward(self, image):
        if torch.rand(1).item() < self.p:
            alpha = torch.empty([image.shape[0], 1, 1, 1], device=image.device).uniform_(self.lower, self.upper)
            image *= alpha
        return image


class ConvertColor(torch.nn.Module):
    def __init__(self, current='RGB', transform='HSV'):
        super(ConvertColor, self).__init__()
        self.current = current
        self.transform = transform

    def forward(self, image):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = F._rgb2hsv(image)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = F._hsv2rgb(image)
        else:
            raise NotImplementedError
        return image


class RandomSaturation(torch.nn.Module):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        super(RandomSaturation, self).__init__()
        assert lower <= upper, f"saturation upper({upper:.2f}) must be larger than lower({lower:.2f})."
        assert 0 <= lower, f"saturation lower({lower:.2f}) must be non-negative."
        assert 0 <= p <= 1, f"saturation probability({p:.2f}) must be in [0, 1]."
        self.lower = lower
        self.upper = upper
        self.p = p

    # expects float image
    def forward(self, image):
        if torch.rand(1).item() < self.p:
            image[:, 1, :, :] *= torch.rand([image.shape[0], 1, 1], device=image.device).uniform_(self.lower, self.upper)
        return image


class RandomHue(torch.nn.Module):
    def __init__(self, delta=18.0 / 255.0, p=0.5):
        super(RandomHue, self).__init__()
        assert 0 <= delta <= 1, f"delta({delta:.2f}) must be in [0, 1]."
        assert 0 <= p <= 1, f"probability({p:.2f}) must be in [0, 1]."
        self.delta = delta
        self.p = p

    # expects float image
    def forward(self, image):
        if torch.rand(1).item() < self.p:
            delta = torch.rand([image.shape[0], 1, 1], device=image.device).uniform_(-self.delta, self.delta)
            image[:, 0, :, :] += delta
            image[:, 0, :, :] %= 1
        return image


class RandomBrightness(torch.nn.Module):
    def __init__(self, delta: float = 32 / 255.0, p=0.5):
        super(RandomBrightness, self).__init__()
        assert 0 <= delta <= 1, f"delta({delta:.2f}) must be in [0, 1]."
        assert 0 <= p <= 1, f"probability({p:.2f}) must be in [0, 1]."
        self.delta = delta
        self.p = p

    def forward(self, image):
        if torch.rand(1).item() < self.p:
            # Generate random number for each image
            delta = torch.rand([image.shape[0], 1, 1, 1], device=image.device).uniform_(-self.delta, self.delta)
            image = (image + delta).clip(0, 1)
        return image


class RandomChannelSwap(torch.nn.Module):
    def __init__(self, p=0.5):
        super(RandomChannelSwap, self).__init__()
        assert 0 <= p <= 1, f"probability({p:.2f}) must be in [0, 1]."
        self.p = p

    def forward(self, image):
        if torch.rand(1).item() < self.p:
            image = image[:, torch.randperm(3), :, :]
        return image
