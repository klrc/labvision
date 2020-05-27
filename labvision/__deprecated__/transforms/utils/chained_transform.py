from torchvision.transforms import transforms


class ChainedTransform():

    def __init__(self):
        """
            chained tranform for torchvision.transforms.
            example_1:
                >>> def basic_normalize(**args):
                >>>    x = ChainedTransform()
                >>>    x = x.ToPILImage().ToTensor().Normalize(**args)
                >>>    return x
            example_2:
                >>> transf = ChainedTransform()
                >>> transf = transf.ToPILImage().ToTensor()
        """
        self.transforms = []
        self._compiled_transform = None

    def __call__(self, pic):
        if self._compiled_transform is None:
            self._compile()
        return self._compiled_transform(pic)

    def _add(self, transform):
        self.transforms.append(transform)
        return self

    def _compile(self):
        t = transforms.Compose(self.transforms)
        self._compiled_transform = t
        return t

    def ToTensor(self, **args):
        return self._add(transforms.ToTensor(**args))

    def ToPILImage(self, **args):
        return self._add(transforms.ToPILImage(**args))

    def Normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], **args):
        '''
            default mean & std as ImageNet.
        '''
        return self._add(transforms.Normalize(mean=mean, std=std, **args))

    def Resize(self, **args):
        return self._add(transforms.Resize(**args))

    def Scale(self, **args):
        return self._add(transforms.Scale(**args))

    def CenterCrop(self, **args):
        return self._add(transforms.CenterCrop(**args))

    def Pad(self, **args):
        return self._add(transforms.Pad(**args))

    def Lambda(self, **args):
        return self._add(transforms.Lambda(**args))

    def RandomApply(self, **args):
        return self._add(transforms.RandomApply(**args))

    def RandomChoice(self, **args):
        return self._add(transforms.RandomChoice(**args))

    def RandomOrder(self, **args):
        return self._add(transforms.RandomOrder(**args))

    def RandomCrop(self, **args):
        return self._add(transforms.RandomCrop(**args))

    def RandomHorizontalFlip(self, **args):
        return self._add(transforms.RandomHorizontalFlip(**args))

    def RandomVerticalFlip(self, **args):
        return self._add(transforms.RandomVerticalFlip(**args))

    def RandomResizedCrop(self, **args):
        return self._add(transforms.RandomResizedCrop(**args))

    def RandomSizedCrop(self, **args):
        return self._add(transforms.RandomSizedCrop(**args))

    def FiveCrop(self, **args):
        return self._add(transforms.FiveCrop(**args))

    def TenCrop(self, **args):
        return self._add(transforms.TenCrop(**args))

    def LinearTransformation(self, **args):
        return self._add(transforms.LinearTransformation(**args))

    def ColorJitter(self, **args):
        return self._add(transforms.ColorJitter(**args))

    def RandomRotation(self, **args):
        return self._add(transforms.RandomRotation(**args))

    def RandomAffine(self, **args):
        return self._add(transforms.RandomAffine(**args))

    def Grayscale(self, **args):
        return self._add(transforms.Grayscale(**args))

    def RandomGrayscale(self, **args):
        return self._add(transforms.RandomGrayscale(**args))

    def RandomPerspective(self, **args):
        return self._add(transforms.RandomPerspective(**args))

    def RandomErasing(self, **args):
        return self._add(transforms.RandomErasing(**args))
