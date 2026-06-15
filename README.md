# 深度相机抠像客户端

这是一个面向 Femto Bolt / K4A 兼容深度相机的 RGB-D 抠像客户端。它把 RVM 视觉抠像和深度范围约束结合起来，用于提升固定拍摄距离下的人像分割稳定性。

## 功能

- Femto Bolt / K4A 兼容深度相机采集
- RGB、Depth、Matting 三路预览
- RVM ONNX 模型抠像
- ONNX Runtime DirectML 推理，优先使用 Windows 独立显卡
- 深度范围、厚度、羽化、腿部补偿等参数可调
- 支持开启或关闭 RVM，仅查看深度抠像效果
- 支持背景图片和背景视频
- 保存当前抠像合成 PNG、RGB PNG 和深度预览 PNG

## 环境要求

- Windows 10/11
- .NET 8 SDK
- Femto Bolt / Orbbec SDK 运行环境
- 支持 DirectML 的显卡和驱动

## 模型

请下载 RVM ONNX 模型，并放到：

```text
models\rvm_mobilenetv3_fp32.onnx
```

推荐模型：

```text
https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
```

模型文件较大，默认不提交到 GitHub。

## 启动

在项目目录运行：

```powershell
.\start_depth_client.cmd
```

或者：

```powershell
dotnet run --project .\DepthMattingClient
```

## 使用说明

1. 连接 Femto Bolt 或 K4A 兼容深度相机。
2. 启动客户端，确认 RGB 和 Depth 都有画面。
3. 设置目标距离，例如 3 米拍摄可把目标距离设为 `3000 mm`。
4. 调整深度厚度，厚度越大，保留的人体范围越完整，但背景误保留也会增加。
5. 开启 RVM 时，程序会融合视觉抠像和深度约束。
6. 关闭 RVM 时，可以单独观察深度切片效果，便于判断深度参数是否合适。
7. 点击保存当前帧，输出当前抠像合成结果。

## 3 米拍摄建议

- 目标距离：`3000 mm`
- 深度容差：`1000` 到 `1400 mm`
- 腿部容易断开时，先增大深度厚度，再观察是否引入背景误抠
- 黑色裤子边缘差时，优先改善人物正面补光和地面反光，软件参数只能补偿一部分

## 开源协议

本项目使用 MIT License。第三方依赖、Femto/Orbbec SDK、RVM 模型和相关算法请遵守其原始项目许可证。
