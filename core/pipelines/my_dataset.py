import os
from PIL import Image
from torch.utils.data import Dataset


class UIDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(UIDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "raw"))]
        self.raw_img_list = [os.path.join(data_root, "raw", i) for i in img_names]
        self.ref_img_list = [os.path.join(data_root, "ref", i) for i in img_names]

        # check files 检查对应图像
        for i in self.ref_img_list:
            i = i.split('\\')[-1]
            if i in img_names is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        raw_img = Image.open(self.raw_img_list[idx])
        ref_img = Image.open(self.ref_img_list[idx])

        if self.transforms is not None:
            raw_img, ref_img = self.transforms(raw_img, ref_img)
        return raw_img, ref_img

    def __len__(self):
        return len(self.raw_img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

