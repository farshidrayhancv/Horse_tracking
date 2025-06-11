import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from dlclibrary import download_huggingface_model

# Copy the exact HRNet modules you provided
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
        out = self.conv2(out)  # BUG FIX: was self.conv2(x), should be self.conv2(out)
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
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
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

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
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

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
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
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.final_layer(x[0])

        return x

def load_superanimal_hrnet():
    """Load SuperAnimal HRNet with correct architecture and weights"""
    print("🔄 Loading SuperAnimal HRNet-w32...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Download model
    models_dir = Path("./superanimal_models")
    models_dir.mkdir(exist_ok=True)
    download_huggingface_model("superanimal_quadruped_hrnet_w32", models_dir)
    
    # Load SuperAnimal state dict
    pose_path = models_dir / "superanimal_quadruped_hrnet_w32.pt"
    superanimal_data = torch.load(pose_path, map_location=device, weights_only=False)
    superanimal_state_dict = superanimal_data['model']
    
    # Create HRNet-w32 for 39 keypoints (SuperAnimal format)
    model = HRNet(c=32, nof_joints=39, bn_momentum=0.1)
    
    # Map SuperAnimal state dict to HRNet
    hrnet_state_dict = {}
    
    for key, value in superanimal_state_dict.items():
        if key.startswith('backbone.model.'):
            # Remove 'backbone.model.' prefix
            hrnet_key = key.replace('backbone.model.', '')
            hrnet_state_dict[hrnet_key] = value
        elif key == 'heads.bodypart.heatmap_head.deconv_layers.0.weight':
            # SuperAnimal: [32, 39, 1, 1] → HRNet: [39, 32, 1, 1]
            # Transpose the first two dimensions
            transposed_weight = value.transpose(0, 1)
            hrnet_state_dict['final_layer.weight'] = transposed_weight
            print(f"Transposed final layer weight: {value.shape} → {transposed_weight.shape}")
        elif key == 'heads.bodypart.heatmap_head.deconv_layers.0.bias':
            hrnet_state_dict['final_layer.bias'] = value
    
    print(f"Mapped {len(hrnet_state_dict)} SuperAnimal parameters to HRNet")
    
    # Load the weights
    missing_keys, unexpected_keys = model.load_state_dict(hrnet_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    if len(missing_keys) < 20:  # Show if not too many
        print(f"Missing: {missing_keys}")
    
    model.to(device)
    model.eval()
    
    print("✅ SuperAnimal HRNet-w32 loaded successfully!")
    print(f"   Architecture: HRNet(c=32, nof_joints=39)")
    print(f"   Device: {device}")
    
    return model, device

def test_superanimal_hrnet():
    """Test SuperAnimal HRNet on test_horse.jpg"""
    print("🧪 Testing Real SuperAnimal HRNet-w32")
    print("=" * 50)
    
    # Load model
    model, device = load_superanimal_hrnet()
    
    # Load test image
    image_path = "test_horse.jpg"
    if not Path(image_path).exists():
        print(f"❌ {image_path} not found!")
        return
    
    image = cv2.imread(image_path)
    print(f"📸 Loaded image: {image.shape}")
    
    # Preprocess
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (192, 256))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0).to(device)
    print(f"🔄 Input tensor: {input_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        heatmaps = model(input_tensor)
    
    print(f"✅ HRNet inference successful!")
    print(f"  Output shape: {heatmaps.shape}")
    print(f"  Expected: (1, 39, H, W)")
    
    # Convert to keypoints
    keypoints = heatmaps_to_keypoints(heatmaps, image.shape)
    
    if keypoints is not None:
        print(f"✅ Generated {len(keypoints)} keypoints")
        print(f"  Confidence range: {keypoints[:, 2].min():.3f} - {keypoints[:, 2].max():.3f}")
        
        # Visualize
        visualize_horse_pose(image, keypoints)
        
    print("\n" + "=" * 50)
    print("🎉 Real SuperAnimal HRNet-w32 test complete!")

def heatmaps_to_keypoints(heatmaps, original_shape):
    """Convert heatmaps to keypoints"""
    heatmaps_np = heatmaps.cpu().numpy().squeeze(0)  # Remove batch dim
    
    keypoints = []
    orig_h, orig_w = original_shape[:2]
    heatmap_h, heatmap_w = heatmaps_np.shape[1], heatmaps_np.shape[2]
    
    for i in range(39):
        heatmap = heatmaps_np[i]
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Scale to original image
        x_coord = (x_idx / heatmap_w) * orig_w
        y_coord = (y_idx / heatmap_h) * orig_h
        confidence = float(heatmap[y_idx, x_idx])
        
        keypoints.append([x_coord, y_coord, confidence])
    
    return np.array(keypoints)

def visualize_horse_pose(image, keypoints):
    """Visualize 39-keypoint horse pose"""
    result = image.copy()
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1:
            cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(result, str(i), (int(x)+5, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Horse skeleton
    skeleton = [
        (0, 5), (0, 10),  # nose to eyes
        (5, 6), (10, 11),  # eyes to ears  
        (15, 16), (17, 18),  # neck
        (19, 20), (20, 21), (21, 22), (22, 23),  # spine to tail
        (15, 24), (24, 25), (25, 26),  # left front leg
        (15, 27), (27, 28), (28, 29),  # right front leg
        (22, 31), (31, 33), (33, 30),  # left back leg
        (22, 32), (32, 34), (34, 35),  # right back leg
    ]
    
    for start_idx, end_idx in skeleton:
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > 0.1 and keypoints[end_idx][2] > 0.1):
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(result, start_point, end_point, (255, 0, 0), 2)
    
    cv2.imwrite("real_superanimal_result.jpg", result)
    print("✅ Real SuperAnimal visualization saved to real_superanimal_result.jpg")

if __name__ == "__main__":
    test_superanimal_hrnet()
