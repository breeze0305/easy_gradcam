import torch
import torch.nn.functional as F
import numpy as np
import math

class EasyGradCAM:
    """
    visualize for classification
    """

    def __init__(
            self, 
            model, 
            targets: list, # List[str] layers structure name, e.g. layer4.2.fc
            device: str = "cuda:0",
            verbose: bool = False,
        ):
        if isinstance(targets, str):
            targets = [targets]
        elif isinstance(targets, list):
            pass
        else:
            raise TypeError("targets must be string or list.")

        assert len(targets) > 0, "please provide targets module name."
        self.verbose = verbose
        self.targets = targets

        self.grads = {}
        self.feats = {}
        for name in targets:
            self.grads[name] = None
            self.feats[name] = None

        self.model = model
        self.device = device
        if "cuda" in device and not torch.cuda.is_available():
            self.device = "cpu"

        self.fw_hooks = []
        self.bw_hooks = []
        for t in targets:
            h = self.get_module_by_name(self.model, t).register_forward_hook(
                    self._make_fw_hook(t)
                )
            self.fw_hooks.append(h)
            h = self.get_module_by_name(self.model, t).register_full_backward_hook(
                    self._make_bw_hook(t)
                )
            self.bw_hooks.append(h)


    def cal_feat_and_grad(
            self, 
            x: torch.Tensor, 
            target_cls: torch.Tensor = None
        ):
        """
        input(torch.Tensor):
            x size: [B, ...]
            target_cls: [B]
        return(dict):
            {name}:{torch.Tensor}
        """
        if target_cls is not None:
            assert target_cls.size(0) == x.size(0), "Please provide the label have same batch size with x."
            assert len(target_cls.size()) == 1, "Only accept one dimension tensor. e.g. [0, 3, 5, 6, 1]"

        batch_size = x.size(0)
        batch_ids = torch.arange(batch_size) # [0, 1, 2, ..., batch_size - 1]

        # 1. forward
        y = self.model(x)
        if not isinstance(y, torch.Tensor): # for huggingface
            y = y.logits
        pred = torch.max(y, dim=1)[1]

        # 2. backward
        # 2.1 get target class
        if target_cls is None: # for hugginface
            pred_y = y[batch_ids, pred]
        else:
            pred_y = y[batch_ids, target_cls]

        pred_y.backward()

        # 3. get features and gradient list
        # 'name':batch_tensor -> [{'name': tensor}, {'name': tensor}, ...]
        batch_feats = []
        batch_grads = []
        for _ in range(batch_size):
            batch_grads.append({})
            batch_feats.append({})

        for name in self.grads:
            for i in range(batch_size):
                batch_grads[i][name] = self.grads[name][i]
                batch_feats[i][name] = self.feats[name][i]

        return batch_feats, batch_grads


    def cal_heats(
            self, 
            ori_imgs: list, # List[np.ndarray], from cv2.imread
            feats: list, # List[dict], from self.get_gradient
            grads: list, # List[dict], from self.get_gradient
            resize_to_ori: bool = True
        ):
        """
        This function will return seperate gradient activate map and the mixture(average) of them.
        --> dict[np.array], np.array
        """

        # ===== check input data-type ======
        if isinstance(ori_imgs, list):
            pass
        elif isinstance(ori_imgs, np.ndarray):
            ori_imgs = [ori_imgs]
        else:
            raise TypeError("[ori_imgs] np.ndarray or list[np.ndarray], and make sure len(list) should equal to batch size of gradients you calculate.")

        if isinstance(grads, list):
            pass
        elif isinstance(grads, dict):
            grads = [grads]
        else:
            raise TypeError("[grads] dict or list[dict], and make sure len(list) should equal to batch size of ori_imgs.")

        if isinstance(feats, list):
            pass
        elif isinstance(feats, dict):
            feats = [feats]
        else:
            raise TypeError("[feats] dict or list[dict], and make sure len(list) should equal to batch size of ori_imgs.")

        assert len(feats) == len(grads) == len(ori_imgs)

        # ===== get gradient activation maps ===== 
        heats = []
        for i in range(len(ori_imgs)):
            heat = self.cal_img_heat(ori_imgs[i], feats[i], grads[i], resize_to_ori)
            heats.append(heat)

        return heats


    def is_perfect_square(self, n: int) -> bool:
        if n < 0:
            return False
        root = math.isqrt(n)
        return root * root == n


    def cal_img_heat(
            self, 
            img: np.ndarray, # single image from cv2.imread w/ BGR format
            feats: dict,
            grads: dict,
            resize_to_ori: bool
        ) -> dict:

        heat = {}
        for name in grads:
            feat = feats[name]
            grad = grads[name]

            if len(grad.size()) == 2: # for transform format (e.g. vit)
                # remove cls token
                grad = grad[1:]
                feat = feat[1:]
                assert self.is_perfect_square(grad.size(0)), f"Only support transformer format, [L, C]. \
And L must be perfect squre. {grad.size(0)} is not."
                N = int(grad.size(0) ** 0.5)
                grad = grad.view(N, N, grad.size(-1)).permute(2, 0, 1).contiguous()
                feat = feat.view(N, N, feat.size(-1)).permute(2, 0, 1).contiguous()
            
            weight = grad.mean(dim=(1, 2))

            # weighted sum of activation maps
            cam = torch.relu(torch.sum(weight[:, None, None] * feat, dim=0))
            # normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            if resize_to_ori:
                H, W = img.shape[:2]
                cam = cam[None, None, ...]
                cam = F.interpolate(cam, 
                                    size=(H, W),
                                    mode="bilinear", 
                                    align_corners=False)[0, 0]

            heat[name] = cam.cpu().numpy() # with shape H, W

        return heat


    def get_module_by_name(self, model, module_name: str):
        names = module_name.split(".")
        module = model
        for n in names:
            # If it's an integer (like "2" in "layer4.2"), cast it
            if n.isdigit():
                module = module[int(n)]
            else:
                module = getattr(module, n)
        return module


    def _make_fw_hook(self, name: str):
        def _forward_hook(module, feat_input, feat_output):
            self.feats[name] = feat_output.detach()
        return _forward_hook

    def _make_bw_hook(self, name: str):
        def _full_backward_hook(module, grad_input, grad_output):
            self.grads[name] = grad_output[0] 
        return _full_backward_hook



