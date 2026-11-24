# Physical AI Tools

This repository offers an interface for developing physical AI applications using LeRobot and ROS 2. For detailed usage instructions, please refer to the documentation below.
  - [Documentation for AI Worker](https://ai.robotis.com/)

To clone this repository:
```bash
git clone -b jazzy https://github.com/ROBOTIS-GIT/physical_ai_tools.git --recursive
```

To learn more about the ROS 2 packages for the AI Worker, visit:
  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)

To explore our open-source platforms in a simulation environment, visit:
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)

For usage instructions and demonstrations of the AI Worker, check out:
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)

To access datasets and pre-trained models for our open-source platforms, see:
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)

To use the Docker image for running ROS packages and Physical AI tools with the AI Worker, visit:
  - [Docker Images](https://hub.docker.com/r/robotis/ros/tags)

## NEXTAGE のタスク空間での模倣学習

NEXTAGE を選択し，収録用サーバーを次のように起動します．

```bash
ros2 launch physical_ai_server physical_ai_server_bringup.launch.py action_profile:=task_delta_6d
```

収録開始前に，NEXTAGE の URDF を `/robot_description` で配信します．収録データセット名の末尾には `_task_delta_6d` が付きます．学習画面ではこのデータセットと ACT などのチャンク方策を選択します．推論画面では学習済みの `pretrained_model` を選択します．学習・推論の追加引数は不要です．引数を省略した収録には従来の関節形式を使います．

| 段階 | データの扱い |
| --- | --- |
| 収録 | 実測関節角と指令関節角をそれぞれ順運動学で変換します．胸リンク基準の右手首・左手首を各9成分で保存し，続けて胸・頭の3関節角を rad で保存します．各手首は位置 xyz（m）と回転行列の第1列・第2列を連結した6成分です． |
| 学習 | 各窓の開始時の実測手首姿勢を基準に，各目標を `inverse(T_start) @ T_target` で表します．胸・頭は同じ時刻の実測値との差です．行動の正規化統計もエピソード内の有効な相対目標から求めます． |
| 推論 | チャンク開始時の実測姿勢を30ステップ保持し，各予測を同じ基準から絶対目標へ復元します．各ステップで現在の関節角を初期値として両腕の逆運動学を解きます．6Dの退化や逆運動学の失敗があれば，その指令を配信する前に推論を終了します． |

ACT，SmolVLA，π0，π0-FAST は予測長・実行長ともに30です．Diffusion は U-Net のダウンサンプリングと観測履歴に合わせて予測長を切り上げ，標準設定では32点を予測して30点を実行します．チェックポイントをコピーする際は，`pretrained_model/action_profile.json` を重みと一緒に保存してください．

既定の NEXTAGE 設定での推論出力先は `/fullbody_controller/follow_joint_trajectory_action/recording_trajectory` です．NEXTAGE 側では送信済み目標を観測するためのトピックです．

チャンク開始時の観測を参照する方式と実行長30の根拠は [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/html/2602.23408v2) にあります．
