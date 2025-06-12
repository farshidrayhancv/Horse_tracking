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
    def __init__(self, device: str = "cpu", config=None):
        self.device = device
        self.config = config  # 🔥 Store config reference for confidence thresholds
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
            actual_scores = []
            if len(predictions) > 0:
                pred = predictions[0]
                if 'scores' in pred and 'boxes' in pred:
                    valid_indices = pred['scores'] > conf_threshold
                    if valid_indices.any():
                        valid_boxes = pred['boxes'][valid_indices]
                        valid_scores = pred['scores'][valid_indices]
                        boxes = valid_boxes.cpu().numpy()
                        actual_scores = valid_scores.cpu().numpy()
            
            if sv and len(boxes) > 0:
                return sv.Detections(
                    xyxy=boxes,
                    confidence=actual_scores,  # ← Fixed: Use real scores instead of hardcoded 0.8
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
        """
        🔥 ENHANCED: SuperAnimal pose estimation with focus on main subject per box.
        Each detection box gets exactly one pose estimation focused on the primary subject.
        """
        if not self.pose_model:
            return []
        
        # Extract bounding boxes from detections
        if sv and hasattr(detections, 'xyxy'):
            boxes = detections.xyxy
        else:
            boxes = detections if len(detections) > 0 else []
        
        if len(boxes) == 0:
            return []
        
        poses = []
        # 🔥 Get confidence threshold from config
        conf_threshold = self.config.confidence_horse_pose_superanimal if self.config else 0.3
        
        try:
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                
                # Safety checks for valid bounding box
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Ensure box is within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(x1 + 1, min(x2, frame_w))
                y2 = max(y1 + 1, min(y2, frame_h))
                
                # 🔥 ENHANCED: Focus on center region for main subject
                # Add small padding to focus on the main subject in the center
                box_w, box_h = x2 - x1, y2 - y1
                center_padding_x = int(box_w * 0.05)  # 5% padding
                center_padding_y = int(box_h * 0.05)
                
                focused_x1 = max(x1, x1 + center_padding_x)
                focused_y1 = max(y1, y1 + center_padding_y)
                focused_x2 = min(x2, x2 - center_padding_x)
                focused_y2 = min(y2, y2 - center_padding_y)
                
                # Use focused crop if it's large enough, otherwise use original
                if focused_x2 - focused_x1 > 50 and focused_y2 - focused_y1 > 50:
                    cropped = frame[focused_y1:focused_y2, focused_x1:focused_x2]
                    crop_offset_x, crop_offset_y = focused_x1, focused_y1
                else:
                    cropped = frame[y1:y2, x1:x2]
                    crop_offset_x, crop_offset_y = x1, y1
                
                if cropped.size == 0:
                    continue
                
                crop_h, crop_w = cropped.shape[:2]
                if crop_h < 10 or crop_w < 10:
                    continue
                
                # Preprocess the CROPPED image
                crop_tensor = self.preprocess_crop_pose(cropped)
                
                with torch.no_grad():
                    heatmaps = self.pose_model(crop_tensor)
                
                # Convert heatmaps to keypoints (with proper offset correction)
                keypoints = self.heatmaps_to_keypoints_focused(heatmaps, crop_offset_x, crop_offset_y, crop_w, crop_h)
                
                if keypoints is not None:
                    # 🔥 CRITICAL: Apply SOURCE-LEVEL confidence filtering
                    filtered_keypoints = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for i, kpt in enumerate(keypoints):
                        x_coord, y_coord, confidence = kpt
                        if confidence > conf_threshold:
                            filtered_keypoints.append([x_coord, y_coord, confidence])
                            valid_count += 1
                            total_confidence += confidence
                        else:
                            filtered_keypoints.append([-1.0, -1.0, 0.0])  # Invalid keypoint marker
                    
                    # Calculate average confidence only from valid keypoints
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    # Only accept poses with sufficient valid keypoints
                    if valid_count >= 5:  # At least 5 valid keypoints for a meaningful pose
                        poses.append({
                            'keypoints': np.array(filtered_keypoints),
                            'box': box,
                            'method': 'SuperAnimal',
                            'confidence': avg_confidence,
                            'box_index': box_idx
                        })
                        # print(f"🔥 SuperAnimal Box {box_idx}: {valid_count}/39 keypoints above {conf_threshold}, avg_conf: {avg_confidence:.3f}")
            
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
    
    def heatmaps_to_keypoints_focused(self, heatmaps, offset_x, offset_y, crop_w, crop_h):
        """Convert heatmaps to keypoints with proper coordinate mapping and sigmoid normalization"""
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
            
            for i in range(min(39, heatmaps.shape[0])):
                heatmap = heatmaps[i]
                y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # Map from heatmap coordinates to crop coordinates, then to original image coordinates
                x_coord = offset_x + (x_idx / heatmap.shape[1]) * crop_w
                y_coord = offset_y + (y_idx / heatmap.shape[0]) * crop_h
                
                # ← Fixed: Normalize confidence to 0-1 range using sigmoid
                raw_confidence = float(heatmap[y_idx, x_idx])
                confidence = 1.0 / (1.0 + np.exp(-raw_confidence))
                
                keypoints.append([x_coord, y_coord, confidence])
            
            return np.array(keypoints)
        except Exception as e:
            print(f"Error converting heatmaps: {e}")
            return None
    
    def heatmaps_to_keypoints(self, heatmaps, box):
        """Legacy method - use heatmaps_to_keypoints_focused for new code"""
        x1, y1, x2, y2 = box
        return self.heatmaps_to_keypoints_focused(heatmaps, x1, y1, x2-x1, y2-y1)