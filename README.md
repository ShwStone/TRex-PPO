# 使用 PPO 进行 TRex 小恐龙游戏

![](./images/best.gif)

上面的结果仅在一张 RTX 3090 上训练了半个小时得到，PPO 的效果远超 [DQN](https://github.com/ShwStone/TRex-DQN) 的效果。

本项目还有许多处理不完善的地方，但是作为了解 PPO 的入门项目，无疑是简单而有趣的。

## 安装依赖项

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

模型文件输出在 `models/` 目录下。

游戏记录输出在 `record/` 目录下。

## 致谢

游戏部分基于此项目进行改编：[SigureMo/T-Rex-runner-pygame: :t-rex: T-rex running implemented with pygame](https://github.com/SigureMo/T-Rex-runner-pygame/)。在其基础上关闭显示，实现了逐帧控制。

PPO 的框架改编自 [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL/)，针对较长的游戏周期，实现了 mini-batch 训练。