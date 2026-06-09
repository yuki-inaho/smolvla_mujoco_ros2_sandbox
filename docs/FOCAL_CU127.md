# focal-cu127 ブランチ: Ubuntu 20.04 (Focal) + CUDA 12.7 対応

upstream（`main`）は Ubuntu 22.04/24.04 を前提にしていますが、本ブランチは
実機の開発マシン環境（Ubuntu 20.04 Focal + CUDA 12.7）に合わせた調整を入れます。

## 対象環境（検証マシンの実測値）

| 項目 | 値 |
|---|---|
| OS | Ubuntu 20.04.6 LTS (Focal), kernel 6.8.0-52-generic, `linux-64` |
| GPU | NVIDIA RTX A5000 ×1 / VRAM 24GB / Ampere (Compute Capability 8.6) |
| ドライバ / CUDA | Driver 565.57.01 / CUDA 12.7 |
| CPU | Intel Xeon Gold 6342 @ 2.80GHz / 96 論理コア |
| メモリ | 503GiB |

GPU は会話ログの「快適ライン（24GB）」を満たすため、SmolVLA 推論・fine-tune・
MuJoCo レンダリング・ROS ノード同時実行を 1 枚でこなせます。

## なぜ Focal (20.04) でも動くか

- ROS 2 Humble は **RoboStack（`robostack-humble` + `conda-forge`）経由で pixi 環境内に
  閉じ込める**ため、ホスト OS の `/opt/ros` に依存しません。Humble の公式対応は 22.04 ですが、
  本構成はシステム ROS を使わないので Focal ホストでも問題ありません（`docs/UBUNTU_NOTES.md` 参照）。
- MuJoCo のヘッドレス描画は `MUJOCO_GL=egl` を使用します。`scripts/activate_ros_overlay.sh`
  が既定で `egl` を設定済みで、NVIDIA driver 565 で EGL が利用できます。

## CUDA 12.7 と PyTorch

- `pixi.toml` に `[system-requirements] cuda = "12.7"` を宣言しました。
  これによりソルバが GPU 対応ビルドを選択でき、ドライバ対応上限を超える要求があれば早期に失敗します。
- PyTorch は `lerobot[smolvla]` 経由で PyPI から入ります。近年の torch wheel は CUDA 12.x ランタイムを
  同梱しており、driver/CUDA 12.7 と前方互換で動作します。
- 導入後は必ず `pixi run doctor` で `torch=..., cuda_available=True, device=NVIDIA RTX A5000` を確認してください。

## 運用上の注意（A5000 24GB）

- `scripts/train_smolvla.sh` の既定 `BATCH_SIZE=64` / `STEPS=20000` は公式 A100 例に準拠した値です。
  A5000 24GB では batch 64 が VRAM 的に厳しい場合があります。まず小さめで疎通を取ってください。

  ```bash
  export HF_DATASET_REPO_ID=my-org/my-robot-dataset
  BATCH_SIZE=16 STEPS=2000 pixi run train-smolvla   # 疎通用の小さい設定例
  ```

- MuJoCo 描画プロセスと SmolVLA 推論プロセスはトピックで分離済みのため、
  24GB あれば描画と推論の同居 OOM は起きにくい構成です。

## このブランチでの変更点

- `pixi.toml`: `[system-requirements] cuda = "12.7"` を追加。
- `docs/FOCAL_CU127.md`（本書）を追加。
- `README.md`: 前提環境に Focal 対応の注記と本書への参照を追加。
