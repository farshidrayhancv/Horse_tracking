import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from dlclibrary import download_huggingface_model
    SUPERANIMAL_AVAILABLE = True
except ImportError:
    SUPERANIMAL_AVAILABLE = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused

class HRNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 3 (stage3)
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 4 (stage4)
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]

        x = self.stage4(x)
        x = self.final_layer(x[0])
        return x

class SuperAnimalQuadruped:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector_model = None
        self.pose_model = None
        self.setup_models()
        self.setup_keypoints()
    
    def setup_models(self):
        models_dir = Path("./superanimal_models")
        models_dir.mkdir(exist_ok=True)
        
        print("🔄 Setting up SuperAnimal-Quadruped models using DLClibrary...")
        
        if not SUPERANIMAL_AVAILABLE:
            print("❌ DLClibrary not available - install with: pip install dlclibrary")
            return
        
        try:
            detector_path = models_dir / "superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt"
            pose_path = models_dir / "superanimal_quadruped_hrnet_w32.pt"
            
            if detector_path.exists() and pose_path.exists():
                print("✅ Models found locally")
                
                detector_data = torch.load(detector_path, map_location=self.device, weights_only=False)
                pose_data = torch.load(pose_path, map_location=self.device, weights_only=False)
                
                self.detector_model = self.build_detector_from_state_dict(detector_data)
                self.pose_model = self.build_pose_from_state_dict(pose_data)
                
                if self.detector_model:
                    print("✅ SuperAnimal detector ready")
                if self.pose_model:
                    print("✅ SuperAnimal pose model ready")
            else:
                print("❌ Model files not found after download")
                
        except Exception as e:
            print(f"❌ Failed to setup SuperAnimal models: {e}")
                
    def build_detector_from_state_dict(self, state_dict_data):
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            
            model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=91)
            
            if isinstance(state_dict_data, dict):
                if 'model' in state_dict_data:
                    state_dict = state_dict_data['model']
                elif 'state_dict' in state_dict_data:
                    state_dict = state_dict_data['state_dict']
                else:
                    state_dict = state_dict_data
            else:
                state_dict = state_dict_data
            
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Failed to build detector: {e}")
            return None
    
    def build_pose_from_state_dict(self, state_dict_data):
        try:
            if 'model' in state_dict_data:
                weights = state_dict_data['model']
            else:
                weights = state_dict_data
            
            final_weight_key = 'heads.bodypart.heatmap_head.deconv_layers.0.weight'
            if final_weight_key not in weights:
                print("❌ SuperAnimal final layer not found")
                return None
            
            model = HRNet(c=32, nof_joints=39, bn_momentum=0.1)
            hrnet_state_dict = {}
            
            for key, value in weights.items():
                if key.startswith('backbone.model.'):
                    hrnet_key = key.replace('backbone.model.', '')
                    hrnet_state_dict[hrnet_key] = value
                elif key == 'heads.bodypart.heatmap_head.deconv_layers.0.weight':
                    transposed_weight = value.transpose(0, 1)
                    hrnet_state_dict['final_layer.weight'] = transposed_weight
                elif key == 'heads.bodypart.heatmap_head.deconv_layers.0.bias':
                    hrnet_state_dict['final_layer.bias'] = value
            
            model.load_state_dict(hrnet_state_dict, strict=False)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Failed to build Real HRNet pose model: {e}")
            return None
    
    def setup_keypoints(self):
        self.keypoint_names = [
            'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
            'right_eye', 'right_earbase', 'right_earend', 'right_antler_base', 'right_antler_end',
            'left_eye', 'left_earbase', 'left_earend', 'left_antler_base', 'left_antler_end',
            'neck_base', 'neck_end', 'throat_base', 'throat_end', 'back_base',
            'back_end', 'back_middle', 'tail_base', 'tail_end', 'front_left_thai',
            'front_left_knee', 'front_left_paw', 'front_right_thai', 'front_right_knee', 'front_right_paw',
            'back_left_paw', 'back_left_thai', 'back_right_thai', 'back_left_knee', 'back_right_knee',
            'back_right_paw', 'belly_bottom', 'body_middle_right', 'body_middle_left'
        ]
        
        self.skeleton = [
            (0, 1), (1, 2), (0, 5), (0, 10),
            (5, 6), (6, 7), (10, 11), (11, 12),
            (15, 16), (17, 18), (19, 20), (20, 21), (21, 22), (22, 23),
            (15, 24), (24, 25), (25, 26),
            (15, 27), (27, 28), (28, 29),
            (22, 31), (31, 33), (33, 30),
            (22, 32), (32, 34), (34, 35),
            (36, 37), (36, 38),
        ]
    
    def detect_quadrupeds(self, frame: np.ndarray, confidence: float = None):
        if not self.detector_model:
            return sv.Detections.empty() if sv else []
        
        conf_threshold = confidence if confidence is not None else 0.5
        
        try:
            frame_tensor = self.preprocess_image_detection(frame)
            with torch.no_grad():
                predictions = self.detector_model(frame_tensor)
            
            boxes = []
            if len(predictions) > 0:
                pred = predictions[0]
                if 'scores' in pred and 'boxes' in pred:
                    valid_indices = pred['scores'] > conf_threshold
                    if valid_indices.any():
                        valid_boxes = pred['boxes'][valid_indices]
                        boxes = valid_boxes.cpu().numpy()
            
            if sv and len(boxes) > 0:
                return sv.Detections(
                    xyxy=boxes,
                    confidence=np.ones(len(boxes)) * 0.8,
                    class_id=np.ones(len(boxes), dtype=int) * 17
                )
            elif sv:
                return sv.Detections.empty()
            else:
                return boxes
        except Exception as e:
            print(f"Error in SuperAnimal detection: {e}")
            return sv.Detections.empty() if sv else []
    
    def preprocess_image_detection(self, frame: np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        tensor = transform(frame_rgb).unsqueeze(0)
        return tensor.to(self.device)
    
    def estimate_pose(self, frame: np.ndarray, detections):
        if not self.pose_model:
            return []
        
        if sv and hasattr(detections, 'xyxy'):
            boxes = detections.xyxy
        else:
            boxes = detections if len(detections) > 0 else []
        
        if len(boxes) == 0:
            return []
        
        poses = []
        try:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                crop_tensor = self.preprocess_crop_pose(cropped)
                with torch.no_grad():
                    heatmaps = self.pose_model(crop_tensor)
                
                keypoints = self.heatmaps_to_keypoints(heatmaps, box)
                if keypoints is not None:
                    poses.append({
                        'keypoints': keypoints,
                        'box': box,
                        'method': 'SuperAnimal',
                        'confidence': np.mean(keypoints[:, 2])
                    })
            return poses
        except Exception as e:
            print(f"Error in SuperAnimal pose estimation: {e}")
            return []
    
    def preprocess_crop_pose(self, crop: np.ndarray):
        crop_resized = cv2.resize(crop, (192, 256))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(crop_rgb).unsqueeze(0)
        return tensor.to(self.device)
    
    def heatmaps_to_keypoints(self, heatmaps, box):
        try:
            if isinstance(heatmaps, tuple):
                heatmaps = heatmaps[0]
            elif isinstance(heatmaps, dict):
                if 'heatmaps' in heatmaps:
                    heatmaps = heatmaps['heatmaps']
                elif 'output' in heatmaps:
                    heatmaps = heatmaps['output']
                else:
                    heatmaps = list(heatmaps.values())[0]
            
            if hasattr(heatmaps, 'cpu'):
                heatmaps = heatmaps.cpu().numpy()
            
            if heatmaps.ndim == 4:
                heatmaps = heatmaps.squeeze(0)
            
            if heatmaps.ndim != 3:
                return None
            
            keypoints = []
            x1, y1, x2, y2 = box
            
            for i in range(min(39, heatmaps.shape[0])):
                heatmap = heatmaps[i]
                y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                x_coord = x1 + (x_idx / heatmap.shape[1]) * (x2 - x1)
                y_coord = y1 + (y_idx / heatmap.shape[0]) * (y2 - y1)
                confidence = heatmap[y_idx, x_idx]
                keypoints.append([x_coord, y_coord, confidence])
            
            return np.array(keypoints)
        except Exception as e:
            print(f"Error converting heatmaps: {e}")
            return None