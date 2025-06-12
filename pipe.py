import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import yaml
import json
warnings.filterwarnings("ignore")

# Add tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for progress bars: pip install tqdm")

# Official HuggingFace Transformers ViTPose (for humans)
try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    VITPOSE_AVAILABLE = True
    print("✓ Official ViTPose (HuggingFace Transformers) available")
except ImportError:
    VITPOSE_AVAILABLE = False
    print("⚠️ ViTPose not available - install with: pip install transformers torch")

# SuperAnimal-Quadruped (for horses) - Using official DLClibrary
try:
    from dlclibrary import download_huggingface_model
    SUPERANIMAL_AVAILABLE = True
    print("✓ Official DLClibrary + SuperAnimal available")
except ImportError:
    SUPERANIMAL_AVAILABLE = False
    print("⚠️ DLClibrary not available - install with: pip install dlclibrary")

# Supervision for professional visualizations and Detection class
try:
    import supervision as sv
    print("✓ Supervision 0.25.1 for professional visualizations and Detection class")
except ImportError:
    print("⚠️ Supervision not available - install with: pip install supervision")
    sv = None


class Config:
    """Configuration class for choosing models and methods"""
    
    def __init__(self, config_file: str = None):
        # Available detectors
        self.HUMAN_DETECTORS = {
            'rtdetr': 'RT-DETR (HuggingFace)',
            'superanimal': 'SuperAnimal Faster R-CNN (fallback only)'
        }
        
        self.HORSE_DETECTORS = {
            'rtdetr': 'RT-DETR (HuggingFace)', 
            'superanimal': 'SuperAnimal Faster R-CNN',
            'both': 'RT-DETR primary + SuperAnimal fallback'
        }
        
        # Available pose estimators
        self.HUMAN_POSE_ESTIMATORS = {
            'vitpose': 'ViTPose 17 keypoints (HuggingFace)',
            'none': 'No human pose estimation'
        }
        
        self.HORSE_POSE_ESTIMATORS = {
            'superanimal': 'SuperAnimal 39 keypoints only',
            'vitpose': 'ViTPose 17 keypoints only (treats horses as humans)',
            'dual': 'Dual competition: SuperAnimal vs ViTPose (best confidence wins)',
            'none': 'No horse pose estimation'
        }
        
        # Default configuration
        self.human_detector = 'rtdetr'
        self.horse_detector = 'both'  # RT-DETR primary + SuperAnimal fallback
        self.human_pose_estimator = 'vitpose'
        self.horse_pose_estimator = 'dual'  # Competition mode
        
        # Separate confidence thresholds for each model/task
        self.confidence_human_detection = 0.3
        self.confidence_horse_detection = 0.3
        self.confidence_human_pose = 0.3
        self.confidence_horse_pose_superanimal = 0.3
        self.confidence_horse_pose_vitpose = 0.3
        
        # Legacy confidence threshold (for backward compatibility)
        self.confidence_threshold = 0.3
        
        # Other settings
        self.jockey_overlap_threshold = 0.4
        
        # Display and output settings
        self.display = False
        self.max_frames = None
        self.output_path = None
        
        # Device
        self.device = "cpu"
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"⚠️ Config file '{config_file}' not found, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    print(f"⚠️ Unsupported config file format: {config_path.suffix}")
                    return
            
            # Load configuration values
            self.human_detector = config_data.get('human_detector', self.human_detector)
            self.horse_detector = config_data.get('horse_detector', self.horse_detector)
            self.human_pose_estimator = config_data.get('human_pose_estimator', self.human_pose_estimator)
            self.horse_pose_estimator = config_data.get('horse_pose_estimator', self.horse_pose_estimator)
            
            # Load confidence thresholds
            self.confidence_human_detection = config_data.get('confidence_human_detection', self.confidence_human_detection)
            self.confidence_horse_detection = config_data.get('confidence_horse_detection', self.confidence_horse_detection)
            self.confidence_human_pose = config_data.get('confidence_human_pose', self.confidence_human_pose)
            self.confidence_horse_pose_superanimal = config_data.get('confidence_horse_pose_superanimal', self.confidence_horse_pose_superanimal)
            self.confidence_horse_pose_vitpose = config_data.get('confidence_horse_pose_vitpose', self.confidence_horse_pose_vitpose)
            
            # Backward compatibility - if confidence_threshold is set, apply to all
            if 'confidence_threshold' in config_data:
                conf_val = config_data['confidence_threshold']
                self.confidence_human_detection = conf_val
                self.confidence_horse_detection = conf_val
                self.confidence_human_pose = conf_val
                self.confidence_horse_pose_superanimal = conf_val
                self.confidence_horse_pose_vitpose = conf_val
                self.confidence_threshold = conf_val
            
            # Load other settings
            self.jockey_overlap_threshold = config_data.get('jockey_overlap_threshold', self.jockey_overlap_threshold)
            self.display = config_data.get('display', self.display)
            self.max_frames = config_data.get('max_frames', self.max_frames)
            self.output_path = config_data.get('output_path', self.output_path)
            self.device = config_data.get('device', self.device)
            
            print(f"✅ Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            print("Using default configuration")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file"""
        config_data = {
            'human_detector': self.human_detector,
            'horse_detector': self.horse_detector,
            'human_pose_estimator': self.human_pose_estimator,
            'horse_pose_estimator': self.horse_pose_estimator,
            'confidence_human_detection': self.confidence_human_detection,
            'confidence_horse_detection': self.confidence_horse_detection,
            'confidence_human_pose': self.confidence_human_pose,
            'confidence_horse_pose_superanimal': self.confidence_horse_pose_superanimal,
            'confidence_horse_pose_vitpose': self.confidence_horse_pose_vitpose,
            'jockey_overlap_threshold': self.jockey_overlap_threshold,
            'display': self.display,
            'max_frames': self.max_frames,
            'output_path': self.output_path,
            'device': self.device
        }
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving config file: {e}")
    
    def create_default_config_file(self, config_file: str = "pose_config.yaml"):
        """Create a default configuration file"""
        config_data = {
            '# Pose Estimation Configuration': None,
            'human_detector': 'rtdetr',  # rtdetr, superanimal
            'horse_detector': 'both',    # rtdetr, superanimal, both
            'human_pose_estimator': 'vitpose',  # vitpose, none
            'horse_pose_estimator': 'dual',     # superanimal, vitpose, dual, none
            
            '# Confidence Thresholds (0.0 - 1.0)': None,
            'confidence_human_detection': 0.3,
            'confidence_horse_detection': 0.3,
            'confidence_human_pose': 0.3,
            'confidence_horse_pose_superanimal': 0.3,
            'confidence_horse_pose_vitpose': 0.3,
            
            '# Other Settings': None,
            'jockey_overlap_threshold': 0.4,
            'display': False,
            'max_frames': None,
            'output_path': None,
            'device': 'cuda'  # cuda, cpu
        }
        
        # Remove comment keys for actual saving
        clean_config = {k: v for k, v in config_data.items() if not k.startswith('#')}
        
        try:
            with open(config_file, 'w') as f:
                f.write("# Pose Estimation Configuration File\n")
                f.write("# Available options:\n")
                f.write("#   human_detector: rtdetr, superanimal\n")
                f.write("#   horse_detector: rtdetr, superanimal, both\n")
                f.write("#   human_pose_estimator: vitpose, none\n")
                f.write("#   horse_pose_estimator: superanimal, vitpose, dual, none\n")
                f.write("#   device: cuda, cpu\n\n")
                yaml.dump(clean_config, f, default_flow_style=False, indent=2)
            print(f"✅ Default configuration file created: {config_file}")
        except Exception as e:
            print(f"❌ Error creating config file: {e}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for all models (backward compatibility)"""
        self.confidence_human_detection = threshold
        self.confidence_horse_detection = threshold
        self.confidence_human_pose = threshold
        self.confidence_horse_pose_superanimal = threshold
        self.confidence_horse_pose_vitpose = threshold
        self.confidence_threshold = threshold
    
    def set_human_detector(self, detector: str):
        """Set human detector method"""
        if detector in self.HUMAN_DETECTORS:
            self.human_detector = detector
            print(f"✅ Human detector: {self.HUMAN_DETECTORS[detector]}")
        else:
            available = list(self.HUMAN_DETECTORS.keys())
            print(f"❌ Invalid human detector. Available: {available}")
    
    def set_horse_detector(self, detector: str):
        """Set horse detector method"""
        if detector in self.HORSE_DETECTORS:
            self.horse_detector = detector
            print(f"✅ Horse detector: {self.HORSE_DETECTORS[detector]}")
        else:
            available = list(self.HORSE_DETECTORS.keys())
            print(f"❌ Invalid horse detector. Available: {available}")
    
    def set_human_pose_estimator(self, estimator: str):
        """Set human pose estimation method"""
        if estimator in self.HUMAN_POSE_ESTIMATORS:
            self.human_pose_estimator = estimator
            print(f"✅ Human pose estimator: {self.HUMAN_POSE_ESTIMATORS[estimator]}")
        else:
            available = list(self.HUMAN_POSE_ESTIMATORS.keys())
            print(f"❌ Invalid human pose estimator. Available: {available}")
    
    def set_horse_pose_estimator(self, estimator: str):
        """Set horse pose estimation method"""
        if estimator in self.HORSE_POSE_ESTIMATORS:
            self.horse_pose_estimator = estimator
            print(f"✅ Horse pose estimator: {self.HORSE_POSE_ESTIMATORS[estimator]}")
        else:
            available = list(self.HORSE_POSE_ESTIMATORS.keys())
            print(f"❌ Invalid horse pose estimator. Available: {available}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n🔧 Current Configuration:")
        print(f"   Human detector: {self.HUMAN_DETECTORS[self.human_detector]}")
        print(f"   Horse detector: {self.HORSE_DETECTORS[self.horse_detector]}")
        print(f"   Human pose: {self.HUMAN_POSE_ESTIMATORS[self.human_pose_estimator]}")
        print(f"   Horse pose: {self.HORSE_POSE_ESTIMATORS[self.horse_pose_estimator]}")
        print(f"   Confidence - Human detection: {self.confidence_human_detection}")
        print(f"   Confidence - Horse detection: {self.confidence_horse_detection}")
        print(f"   Confidence - Human pose: {self.confidence_human_pose}")
        print(f"   Confidence - Horse pose (SuperAnimal): {self.confidence_horse_pose_superanimal}")
        print(f"   Confidence - Horse pose (ViTPose): {self.confidence_horse_pose_vitpose}")
        print(f"   Jockey overlap: {self.jockey_overlap_threshold}")
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")
        if self.output_path:
            print(f"   Output: {self.output_path}")


# Real HRNet Architecture Classes for SuperAnimal
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
        out = self.conv2(out)  # Fixed: was self.conv2(x)
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
    """SuperAnimal-Quadruped pose estimation model for horses"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector_model = None
        self.pose_model = None
        self.setup_models()
        self.setup_keypoints()
    
    def setup_models(self):
        """Setup SuperAnimal-Quadruped models using official DLClibrary"""
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
        """Build Faster R-CNN detector from state dict"""
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
        """Build Real SuperAnimal HRNet-w32 from state dict"""
        try:
            if 'model' in state_dict_data:
                weights = state_dict_data['model']
            else:
                weights = state_dict_data
            
            final_weight_key = 'heads.bodypart.heatmap_head.deconv_layers.0.weight'
            final_bias_key = 'heads.bodypart.heatmap_head.deconv_layers.0.bias'
            
            if final_weight_key not in weights:
                print("❌ SuperAnimal final layer not found")
                return None
            
            # Create Real HRNet-w32 for 39 keypoints
            model = HRNet(c=32, nof_joints=39, bn_momentum=0.1)
            
            # Map SuperAnimal state dict to HRNet
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
        """Define 39-keypoint SuperAnimal skeleton for quadrupeds"""
        
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
        
        # Skeleton connections for quadruped anatomy
        self.skeleton = [
            # Head connections
            (0, 1), (1, 2), (0, 5), (0, 10),  # nose to jaws and eyes
            (5, 6), (6, 7), (10, 11), (11, 12),  # eyes to ears
            
            # Neck and spine
            (15, 16), (17, 18), (19, 20), (20, 21), (21, 22), (22, 23),  # neck to tail
            
            # Front legs
            (15, 24), (24, 25), (25, 26),  # left front leg
            (15, 27), (27, 28), (28, 29),  # right front leg
            
            # Back legs  
            (22, 31), (31, 33), (33, 30),  # left back leg
            (22, 32), (32, 34), (34, 35),  # right back leg
            
            # Body connections
            (36, 37), (36, 38),  # belly connections
        ]
    
    def detect_quadrupeds(self, frame: np.ndarray, confidence: float = None):
        """Detect quadrupeds using SuperAnimal Faster R-CNN - Returns supervision Detection"""
        if not self.detector_model:
            return sv.Detections.empty() if sv else []
        
        # Use provided confidence or default
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
            
            # Convert to supervision Detection format
            if sv and len(boxes) > 0:
                return sv.Detections(
                    xyxy=boxes,
                    confidence=np.ones(len(boxes)) * 0.8,
                    class_id=np.ones(len(boxes), dtype=int) * 17  # Horse class ID
                )
            elif sv:
                return sv.Detections.empty()
            else:
                return boxes
            
        except Exception as e:
            print(f"Error in SuperAnimal detection: {e}")
            return sv.Detections.empty() if sv else []
    
    def preprocess_image_detection(self, frame: np.ndarray):
        """Preprocess image for Faster R-CNN detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        tensor = transform(frame_rgb).unsqueeze(0)
        return tensor.to(self.device)
    
    def estimate_pose(self, frame: np.ndarray, detections):
        """Estimate 39-keypoint poses using SuperAnimal HRNet - Accepts supervision Detections"""
        if not self.pose_model:
            return []
        
        # Handle both supervision Detections and numpy arrays
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
                        'confidence': np.mean(keypoints[:, 2])  # Average confidence
                    })
            
            return poses
            
        except Exception as e:
            print(f"Error in SuperAnimal pose estimation: {e}")
            return []
    
    def preprocess_crop_pose(self, crop: np.ndarray):
        """Preprocess cropped region for HRNet pose estimation"""
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
        """Convert HRNet heatmaps to keypoint coordinates"""
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


class HybridPoseSystem:
    """Configurable pose estimation system with supervision Detection integration"""
    
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames <= 0:
            # For some formats, frame count might not be available
            print("⚠️ Warning: Frame count not available, will process until end of video")
            self.total_frames = float('inf')
        
        # Setup models based on config
        self.setup_detection()
        self.setup_pose_models()
        self.setup_colors()
        
        # Print configuration
        self.config.print_config()
        print(f"🐴🏇 Configurable Pose System ready: {self.total_frames} frames")
    
    def setup_detection(self):
        """Setup object detection based on config"""
        # RT-DETR detector
        self.rtdetr_detector = None
        self.rtdetr_processor = None
        
        if self.config.human_detector == 'rtdetr' or self.config.horse_detector in ['rtdetr', 'both']:
            if VITPOSE_AVAILABLE:
                try:
                    self.rtdetr_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
                    self.rtdetr_detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
                    self.rtdetr_detector.to(self.config.device)
                    print("✅ RT-DETR detector loaded")
                except Exception as e:
                    print(f"⚠️ RT-DETR failed: {e}")
        
        # SuperAnimal detector
        self.superanimal = None
        if self.config.horse_detector in ['superanimal', 'both']:
            self.superanimal = SuperAnimalQuadruped(device=self.config.device)
    
    def setup_pose_models(self):
        """Setup pose estimation models based on config"""
        # ViTPose for humans
        self.vitpose_processor = None
        self.vitpose_model = None
        
        if self.config.human_pose_estimator == 'vitpose':
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model.to(self.config.device)
                    print("✅ ViTPose (humans) loaded")
                except Exception as e:
                    print(f"⚠️ ViTPose failed: {e}")
        
        # ViTPose for horses (if needed)
        self.vitpose_horse_processor = None
        self.vitpose_horse_model = None
        
        if self.config.horse_pose_estimator in ['vitpose', 'dual']:
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_horse_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model.to(self.config.device)
                    print("✅ ViTPose (for horses) loaded")
                except Exception as e:
                    print(f"⚠️ ViTPose for horses failed: {e}")
        
        # Define skeletons
        self.human_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def setup_colors(self):
        """Setup visualization colors"""
        self.human_color = (0, 255, 0)      # Green for jockeys
        self.superanimal_color = (255, 0, 0)  # Red for SuperAnimal horses
        self.vitpose_horse_color = (255, 0, 255)  # Magenta for ViTPose horses
        self.keypoint_color = (0, 255, 255) # Cyan
        self.text_color = (255, 255, 255)   # White
    
    def detect_humans(self, frame: np.ndarray):
        """Detect humans based on config - Returns supervision Detections"""
        if self.config.human_detector == 'rtdetr' and self.rtdetr_detector:
            return self._detect_rtdetr(frame, class_filter=[0], confidence=self.config.confidence_human_detection)  # Human class
        else:
            return sv.Detections.empty() if sv else []
    
    def detect_horses(self, frame: np.ndarray):
        """Detect horses based on config - Returns supervision Detections"""
        if self.config.horse_detector == 'rtdetr' and self.rtdetr_detector:
            return self._detect_rtdetr(frame, class_filter=[17], confidence=self.config.confidence_horse_detection)  # Horse class
        elif self.config.horse_detector == 'superanimal' and self.superanimal:
            return self.superanimal.detect_quadrupeds(frame, self.config.confidence_horse_detection)
        elif self.config.horse_detector == 'both':
            # Try RT-DETR first, fallback to SuperAnimal
            horse_detections = self._detect_rtdetr(frame, class_filter=[17], confidence=self.config.confidence_horse_detection)
            if sv and len(horse_detections) == 0 and self.superanimal:
                horse_detections = self.superanimal.detect_quadrupeds(frame, self.config.confidence_horse_detection)
            return horse_detections
        else:
            return sv.Detections.empty() if sv else []
    
    def _detect_rtdetr(self, frame: np.ndarray, class_filter: List[int], confidence: float = None):
        """RT-DETR detection helper - Returns supervision Detections"""
        if not self.rtdetr_detector or not self.rtdetr_processor:
            return sv.Detections.empty() if sv else []
        
        # Use provided confidence or default
        conf_threshold = confidence if confidence is not None else self.config.confidence_threshold
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            inputs = self.rtdetr_processor(images=pil_image, return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.rtdetr_detector(**inputs)
            
            results = self.rtdetr_processor.post_process_object_detection(
                outputs, 
                target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
                threshold=conf_threshold
            )
            
            if len(results) > 0:
                result = results[0]
                
                # Filter for desired classes - ensure class_filter tensor is on same device
                class_filter_tensor = torch.tensor(class_filter, device=result["labels"].device)
                class_mask = torch.isin(result["labels"], class_filter_tensor)
                
                if class_mask.any():
                    filtered_boxes = result["boxes"][class_mask].cpu().numpy()
                    filtered_scores = result["scores"][class_mask].cpu().numpy()
                    filtered_labels = result["labels"][class_mask].cpu().numpy()
                    
                    if sv:
                        return sv.Detections(
                            xyxy=filtered_boxes,
                            confidence=filtered_scores,
                            class_id=filtered_labels
                        )
                    else:
                        # Convert to COCO format for legacy support
                        coco_boxes = filtered_boxes.copy()
                        coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
                        coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
                        return coco_boxes
            
            return sv.Detections.empty() if sv else []
            
        except Exception as e:
            print(f"Error in RT-DETR detection: {e}")
            return sv.Detections.empty() if sv else []
    
    def filter_jockeys(self, human_detections, horse_detections):
        """Filter humans to only include jockeys using supervision Detections"""
        if not sv:
            return human_detections
        
        if len(human_detections) == 0 or len(horse_detections) == 0:
            return sv.Detections.empty()
        
        jockey_indices = []
        
        for i, human_box in enumerate(human_detections.xyxy):
            hx1, hy1, hx2, hy2 = human_box
            hw, hh = hx2 - hx1, hy2 - hy1
            
            for horse_box in horse_detections.xyxy:
                rx1, ry1, rx2, ry2 = horse_box
                
                # Calculate intersection
                ix1 = max(hx1, rx1)
                iy1 = max(hy1, ry1)
                ix2 = min(hx2, rx2)
                iy2 = min(hy2, ry2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    human_area = hw * hh
                    overlap_ratio = intersection_area / human_area if human_area > 0 else 0
                    
                    if overlap_ratio >= self.config.jockey_overlap_threshold:
                        jockey_indices.append(i)
                        break
        
        if jockey_indices:
            return sv.Detections(
                xyxy=human_detections.xyxy[jockey_indices],
                confidence=human_detections.confidence[jockey_indices],
                class_id=human_detections.class_id[jockey_indices]
            )
        else:
            return sv.Detections.empty()
    
    def estimate_human_poses(self, frame: np.ndarray, detections):
        """Estimate human poses based on config"""
        if self.config.human_pose_estimator == 'none':
            return []
        elif self.config.human_pose_estimator == 'vitpose':
            return self._estimate_vitpose_human(frame, detections)
        else:
            return []
    
    def estimate_horse_poses(self, frame: np.ndarray, detections):
        """Estimate horse poses based on config"""
        if self.config.horse_pose_estimator == 'none':
            return []
        elif self.config.horse_pose_estimator == 'superanimal':
            return self._estimate_superanimal_only(frame, detections)
        elif self.config.horse_pose_estimator == 'vitpose':
            return self._estimate_vitpose_horse_only(frame, detections)
        elif self.config.horse_pose_estimator == 'dual':
            return self._estimate_dual_competition(frame, detections)
        else:
            return []
    
    def _estimate_vitpose_human(self, frame: np.ndarray, detections):
        """ViTPose human pose estimation"""
        if not self.vitpose_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to COCO format for ViTPose
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_model(**inputs)
            
            pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            return pose_results[0] if pose_results else []
            
        except Exception as e:
            print(f"Error in human pose estimation: {e}")
            return []
    
    def _estimate_superanimal_only(self, frame: np.ndarray, detections):
        """SuperAnimal-only horse pose estimation"""
        if not self.superanimal:
            return []
        
        return self.superanimal.estimate_pose(frame, detections)
    
    def _estimate_vitpose_horse_only(self, frame: np.ndarray, detections):
        """ViTPose-only horse pose estimation"""
        if not self.vitpose_horse_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to COCO format
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_horse_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_horse_model(**inputs)
            
            pose_results = self.vitpose_horse_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            
            if pose_results and len(pose_results[0]) > 0:
                converted_poses = []
                for i, pose_result in enumerate(pose_results[0]):
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # Convert to our format
                    kpts_with_conf = []
                    for kpt, score in zip(keypoints, scores):
                        kpts_with_conf.append([kpt[0], kpt[1], score])
                    
                    converted_poses.append({
                        'keypoints': np.array(kpts_with_conf),
                        'box': detections.xyxy[i],
                        'method': 'ViTPose',
                        'confidence': np.mean(scores)
                    })
                
                return converted_poses
            
            return []
            
        except Exception as e:
            print(f"Error in ViTPose horse estimation: {e}")
            return []
    
    def _estimate_dual_competition(self, frame: np.ndarray, detections):
        """Dual competition: SuperAnimal vs ViTPose for horses"""
        if not sv or len(detections) == 0:
            return []
        
        # Get poses from both methods
        superanimal_poses = self._estimate_superanimal_only(frame, detections)
        vitpose_poses = self._estimate_vitpose_horse_only(frame, detections)
        
        # Competition: Pick best pose per horse based on confidence
        best_poses = []
        num_horses = len(detections)
        
        for i in range(num_horses):
            superanimal_conf = 0.0
            vitpose_conf = 0.0
            
            # Get SuperAnimal confidence
            if i < len(superanimal_poses):
                superanimal_conf = superanimal_poses[i]['confidence']
            
            # Get ViTPose confidence
            if i < len(vitpose_poses):
                vitpose_conf = vitpose_poses[i]['confidence']
            
            # Pick the winner
            if superanimal_conf > vitpose_conf and i < len(superanimal_poses):
                best_poses.append(superanimal_poses[i])
            elif vitpose_conf > 0 and i < len(vitpose_poses):
                best_poses.append(vitpose_poses[i])
        
        return best_poses
    
    def draw_human_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw human pose with 17 keypoints"""
        if 'keypoints' not in pose_result or 'scores' not in pose_result:
            return frame
        
        # Use provided confidence or config value
        conf_threshold = min_confidence if min_confidence is not None else self.config.confidence_human_pose
        
        keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
        scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
        
        # Draw skeleton
        for start_idx, end_idx in self.human_skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                if scores[start_idx] > conf_threshold and scores[end_idx] > conf_threshold:
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, self.human_color, 2)
        
        # Draw keypoints
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > conf_threshold:
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, self.human_color, 1)
        
        return frame
    
    def draw_horse_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw horse pose (SuperAnimal=39kp red, ViTPose=17kp magenta)"""
        if 'keypoints' not in pose_result or 'method' not in pose_result:
            return frame
        
        keypoints = pose_result['keypoints']
        method = pose_result['method']
        
        # Use appropriate confidence threshold based on method
        if min_confidence is not None:
            conf_threshold = min_confidence
        elif method == 'SuperAnimal':
            conf_threshold = self.config.confidence_horse_pose_superanimal
        else:  # ViTPose
            conf_threshold = self.config.confidence_horse_pose_vitpose
        
        # Choose color and skeleton based on method
        if method == 'SuperAnimal':
            color = self.superanimal_color
            skeleton = self.superanimal.skeleton if self.superanimal else []
        else:  # ViTPose
            color = self.vitpose_horse_color
            skeleton = self.human_skeleton  # Use human skeleton for ViTPose horses
        
        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                if keypoints[start_idx][2] > conf_threshold and keypoints[end_idx][2] > conf_threshold:
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if kpt[2] > conf_threshold:
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, color, 1)
        
        return frame
    
    def annotate_detections(self, frame: np.ndarray, human_detections, horse_detections):
        """Annotate detections using supervision if available"""
        if not sv:
            return frame
        
        try:
            # Human detections (green triangles)
            if len(human_detections) > 0:
                triangle_annotator = sv.TriangleAnnotator(
                    base=15, height=20, color=sv.Color.GREEN
                )
                frame = triangle_annotator.annotate(frame, human_detections)
            
            # Horse detections (red triangles)
            if len(horse_detections) > 0:
                triangle_annotator = sv.TriangleAnnotator(
                    base=15, height=20, color=sv.Color.RED
                )
                frame = triangle_annotator.annotate(frame, horse_detections)
                
        except Exception:
            # Fallback to rectangles
            if len(human_detections) > 0:
                for i, box in enumerate(human_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.human_color, 2)
                    cv2.putText(frame, f"Human {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.human_color, 2)
            
            if len(horse_detections) > 0:
                for i, box in enumerate(horse_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.superanimal_color, 2)
                    cv2.putText(frame, f"Horse {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.superanimal_color, 2)
        
        return frame
    
    def process_video(self):
        """Process video with configurable models and optional real-time display"""
        
        # Determine output path
        if self.config.output_path:
            output_path = self.config.output_path
        else:
            # Create safe output filename
            input_stem = self.video_path.stem if self.video_path.suffix else self.video_path.name
            output_path = str(self.video_path.parent / f"{input_stem}_pose_output.mp4")
        
        print(f"🎬 Processing: {self.video_path}")
        print(f"📤 Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'humans_detected': 0, 'horses_detected': 0,
            'human_poses': 0, 'horse_poses': 0,
            'superanimal_wins': 0, 'vitpose_wins': 0
        }
        
        # Initialize progress bar (only if not displaying)
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing pose estimation", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        # Setup display window
        if self.config.display:
            window_name = "Configurable Pose System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect objects based on config
            human_detections = self.detect_humans(frame)
            horse_detections = self.detect_horses(frame)
            
            # Filter humans to jockeys only
            if sv:
                jockey_detections = self.filter_jockeys(human_detections, horse_detections)
            else:
                jockey_detections = human_detections
            
            stats['humans_detected'] += len(jockey_detections) if sv else len(jockey_detections)
            stats['horses_detected'] += len(horse_detections) if sv else len(horse_detections)
            
            # Estimate poses based on config
            human_poses = self.estimate_human_poses(frame, jockey_detections)
            horse_poses = self.estimate_horse_poses(frame, horse_detections)
            
            stats['human_poses'] += len(human_poses)
            stats['horse_poses'] += len(horse_poses)
            
            # Count method wins for dual mode
            if self.config.horse_pose_estimator == 'dual':
                superanimal_count = sum(1 for pose in horse_poses if pose.get('method') == 'SuperAnimal')
                vitpose_count = sum(1 for pose in horse_poses if pose.get('method') == 'ViTPose')
                stats['superanimal_wins'] += superanimal_count
                stats['vitpose_wins'] += vitpose_count
            
            # Annotate detections
            frame = self.annotate_detections(frame, jockey_detections, horse_detections)
            
            # Draw poses
            for pose_result in human_poses:
                frame = self.draw_human_pose(frame, pose_result)
            
            for pose_result in horse_poses:
                frame = self.draw_horse_pose(frame, pose_result)
                
                # Add method label for horses
                if 'box' in pose_result and 'method' in pose_result:
                    box = pose_result['box']
                    method = pose_result['method']
                    conf = pose_result.get('confidence', 0)
                    x1, y1, x2, y2 = box.astype(int)
                    
                    color = self.superanimal_color if method == 'SuperAnimal' else self.vitpose_horse_color
                    kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                    
                    cv2.putText(frame, f"{method} {conf:.2f} ({kp_count})", 
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add info overlay
            human_count = len(jockey_detections) if sv else len(jockey_detections)
            horse_count = len(horse_detections) if sv else len(horse_detections)
            
            total_display = str(max_frames) if self.total_frames != float('inf') else "∞"
            
            info_lines = [
                f"Frame: {frame_count+1}/{total_display}",
                f"Config: H-Det:{self.config.human_detector} H-Pose:{self.config.human_pose_estimator}",
                f"        Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
                f"Detected - Jockeys:{human_count} Horses:{horse_count}",
                f"Poses - Humans:{len(human_poses)} Horses:{len(horse_poses)}"
            ]
            
            if self.config.horse_pose_estimator == 'dual':
                info_lines.append(f"Competition - SuperAnimal:{stats['superanimal_wins']} ViTPose:{stats['vitpose_wins']}")
            
            if self.config.display:
                info_lines.append(f"Controls: SPACE=Pause Q=Quit {'PAUSED' if paused else ''}")
            
            # Semi-transparent background
            overlay = frame.copy()
            overlay_height = 25 + len(info_lines) * 18
            cv2.rectangle(overlay, (5, 5), (950, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            for i, line in enumerate(info_lines):
                y_pos = 25 + i * 18
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame if requested
            if self.config.display:
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n🛑 User requested quit")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif paused and key != 255:  # Any other key when paused
                    pass  # Continue to next frame
            
            frame_count += 1
            
            # Update progress bar (only if available)
            if pbar:
                pbar.set_postfix_str(f"H:{human_count} Horses:{horse_count}")
                pbar.update(1)
        
        self.cap.release()
        out.release()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if pbar:
            pbar.close()
        
        print(f"✅ Processing complete!")
        print(f"📊 Final Stats:")
        print(f"   Humans detected: {stats['humans_detected']}")
        print(f"   Horses detected: {stats['horses_detected']}")
        print(f"   Human poses: {stats['human_poses']}")
        print(f"   Horse poses: {stats['horse_poses']}")
        
        if self.config.horse_pose_estimator == 'dual':
            print(f"🥊 Competition Results:")
            print(f"   SuperAnimal wins: {stats['superanimal_wins']} (39 keypoints)")
            print(f"   ViTPose wins: {stats['vitpose_wins']} (17 keypoints)")
        
        print(f"🎯 Output: {output_path}")
        
        return output_path


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Configurable Pose Estimation System")
        print("Usage: python pipe.py <video> [output] [options...]")
        print("   or: python pipe.py <video> --output <output_file> [options...]")
        print("   or: python pipe.py <video> --config <config_file> [options...]")
        print("\n🔧 Configuration Options:")
        
        config = Config()
        print("Human Detectors:")
        for key, desc in config.HUMAN_DETECTORS.items():
            print(f"  {key}: {desc}")
        
        print("\nHorse Detectors:")
        for key, desc in config.HORSE_DETECTORS.items():
            print(f"  {key}: {desc}")
        
        print("\nHuman Pose Estimators:")
        for key, desc in config.HUMAN_POSE_ESTIMATORS.items():
            print(f"  {key}: {desc}")
        
        print("\nHorse Pose Estimators:")
        for key, desc in config.HORSE_POSE_ESTIMATORS.items():
            print(f"  {key}: {desc}")
        
        print("\n📝 Example Configurations:")
        print("python pipe.py video.mp4 output.mp4 --human-detector rtdetr --horse-detector both --display")
        print("python pipe.py video.mp4 --config pose_config.yaml")
        print("python pipe.py video.mp4 --confidence 0.1  # Sets all confidence values")
        print("python pipe.py video.mp4 --confidence-human-detection 0.2 --confidence-horse-pose-superanimal 0.4")
        print("python pipe.py --create-config pose_config.yaml  # Create default config file")
        
        print("\n🔧 Config File:")
        print("- Use --config <file> to load settings from YAML/JSON file")
        print("- Use --create-config <file> to generate a default config file")
        print("- Command line arguments override config file settings")
        
        print("\n🚀 Features:")
        print("- Modular detection and pose estimation")
        print("- Separate confidence thresholds for each model")
        print("- Configuration file support (YAML/JSON)")
        print("- Supervision Detection class integration (ready for ByteTrack)")
        print("- Real-time display with controls")
        print("- GPU acceleration support")
        
        print("\nRequired packages:")
        print("pip install transformers torch torchvision pillow dlclibrary supervision tqdm pyyaml")
        sys.exit(1)
    
    # Handle config file creation
    if len(sys.argv) >= 3 and sys.argv[1] == '--create-config':
        config = Config()
        config.create_default_config_file(sys.argv[2])
        sys.exit(0)
    
    # Parse arguments
    video_path = sys.argv[1]
    config = Config()
    
    # Check if second argument is output file (no -- prefix)
    i = 2
    if i < len(sys.argv) and not sys.argv[i].startswith('--'):
        config.output_path = sys.argv[i]
        i += 1
    
    # Parse remaining arguments
    while i < len(sys.argv):
        if sys.argv[i] == '--config' and i + 1 < len(sys.argv):
            config.load_from_file(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            config.output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--human-detector' and i + 1 < len(sys.argv):
            config.set_human_detector(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--horse-detector' and i + 1 < len(sys.argv):
            config.set_horse_detector(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--human-pose' and i + 1 < len(sys.argv):
            config.set_human_pose_estimator(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--horse-pose' and i + 1 < len(sys.argv):
            config.set_horse_pose_estimator(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence' and i + 1 < len(sys.argv):
            config.set_confidence_threshold(float(sys.argv[i + 1]))
            i += 2
        elif sys.argv[i] == '--confidence-human-detection' and i + 1 < len(sys.argv):
            config.confidence_human_detection = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence-horse-detection' and i + 1 < len(sys.argv):
            config.confidence_horse_detection = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence-human-pose' and i + 1 < len(sys.argv):
            config.confidence_human_pose = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence-horse-pose-superanimal' and i + 1 < len(sys.argv):
            config.confidence_horse_pose_superanimal = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence-horse-pose-vitpose' and i + 1 < len(sys.argv):
            config.confidence_horse_pose_vitpose = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--jockey-overlap' and i + 1 < len(sys.argv):
            config.jockey_overlap_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--max-frames' and i + 1 < len(sys.argv):
            config.max_frames = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--device' and i + 1 < len(sys.argv):
            config.device = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--display':
            config.display = True
            i += 1
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            i += 1
    
    # Auto-detect device if not specified
    if config.device == "cpu" and VITPOSE_AVAILABLE and torch.cuda.is_available():
        config.device = "cuda"
    
    print(f"🎬 Input: {video_path}")
    if config.output_path:
        print(f"📤 Output: {config.output_path}")
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video file '{video_path}' does not exist")
        sys.exit(1)
    
    try:
        system = HybridPoseSystem(video_path, config)
        system.process_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install: pip install transformers torch torchvision pillow dlclibrary supervision tqdm pyyaml")
        print("2. Check model downloads completed successfully")
        print("3. Try different confidence: --confidence 0.05")
        print("4. Check video file exists and is readable")
        print("5. Try CPU mode: --device cpu")
        print("6. Make sure video file has proper extension (.mp4, .avi, etc.)")
        print("7. Create config file: --create-config config.yaml")


if __name__ == "__main__":
    main()
