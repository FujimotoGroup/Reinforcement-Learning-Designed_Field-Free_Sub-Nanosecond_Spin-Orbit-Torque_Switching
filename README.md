## 注意
計算結果のデータが10万ファイルくらいあるので普通にcloneすると大変なことになるので以下の手順でcloneする．

1. リポジトリをclone（履歴は取るがファイルはチェックアウトしない）
```bash
git clone --filter=blob:none --no-checkout git@github.com:FujimotoGroup/Reinforcement-Learning-Designed_Field-Free_Sub-Nanosecond_Spin-Orbit-Torque_Switching.git
cd Reinforcement-Learning-Designed_Field-Free_Sub-Nanosecond_Spin-Orbit-Torque_Switching/
```
2. スパースチェックアウトを有効に
```bash
git sparse-checkout init --cone
```
3. パスを指定
```bash
git sparse-checkout set src/
```
4. 明示的に checkout
```bash
git checkout
```
