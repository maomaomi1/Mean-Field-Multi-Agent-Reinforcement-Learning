### 平均场多智能体强化学习

**论文中MF-Q和MF-AC的实现**

#### 示例

- 一个在低温下的20x20 Ising模型示例。
- 一个40x40的战斗游戏网格世界示例，包含128个智能体，其中蓝色的是MFQ，红色的是IL。

### 代码结构

- **main_MFQ_Ising.py**：包含运行基于表格的MFQ的Ising模型的代码。
- **./examples/**：包含Ising模型和战斗游戏的场景（以及模型）。

- **battle.py**：包含运行战斗游戏与训练模型的代码。
- **train_battle.py**：包含训练战斗游戏模型的代码。

### 编译Ising环境并运行

#### 需求

- python==3.6.1
- gym==0.9.2（可能与更高版本兼容）
- matplotlib（如果你想生成Ising模型图形）

### 编译MAgent平台并运行

在运行战斗游戏环境之前，你需要编译它。你可以从以下链接获得更多帮助：MAgent

#### 编译步骤

```bash
cd examples/battle_model
./build.sh
```

#### 在战斗游戏设置下训练模型的步骤

在你的 `~/.bashrc` 或 `~/.zshrc` 中添加python路径：

```bash
vim ~/.zshrc
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
source ~/.zshrc
```

运行训练脚本进行训练（例如，mfac）：

```bash
python3 train_battle.py --algo mfac
```

或者获取帮助：

```bash
python3 train_battle.py --help
```

### 论文引用

如果你觉得这篇论文有帮助，请考虑引用以下文献：

```
@InProceedings{pmlr-v80-yang18d,
  title = 	 {Mean Field Multi-Agent Reinforcement Learning},
  author = 	 {Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {5567--5576},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR}
}
```