# 作業記録 (WORKLOG) — focal-cu127: SmolVLA 推論＋可視化

> 統括(team-lead)が随時更新する作業ログ。findings / struggles / tips を時系列＋テーマで残す。
> 体制: worker(Sonnet=実作業) / worker-viz(Sonnet=描画・可視化) / auditor(Opus=独立検証) / team-lead(統括)。
> 対象機: Ubuntu 20.04 Focal / RTX A5000 24GB / Driver 565 / CUDA 12.7。ブランチ `focal-cu127`(main は素のインポートを温存)。

## 現在地 (Status)
- ✅ M0(環境・mock 閉ループ): 完了・監査 PASS。
- ✅ SmolVLA 推論(standalone): 完了・監査 PASS。
- ✅ 実レンダリング(EGL)基盤: 完了・監査 PASS。
- ✅ toy_arm 指令追従(モデル修正): 完了。
- ✅ skills/ 作成(clean-sim-run / record-and-verify-motion / smolvla-infer-smoke / pixi-env-doctor): 完了。
- 🔄 **SmolVLA 駆動の「見えて連続的に動く」E2E**: 進行中。← 現在のボトルネック。worker の v3/v4 は (a)多重プロセス汚染のフリッカ偽陽性、(b)action_scale=0.3 のままで飽和静止、で却下。**クリーン録画(Publisher==1)＋action_scale↓(0.08)で漸進横断**を再指示中。
- 🔄 justfile / docs/ONBOARDING.md (DoD 追加): 作成中(Workflow)。
- ⏳ 最終コミット＋docs 整合。

### DoD（このフェーズ）
1. SmolVLA 推論が CUDA(A5000)でロード&select_action(済) 2. 実レンダリング(EGL)で publish(済) 3. **クリーン単一インスタンスで SmolVLA 駆動の連続可視運動 MP4**(未) 4. justfile で再現性化(進行中) 5. docs/ONBOARDING.md(進行中) 6. auditor が各項目を独立検証 / main 温存。

---

## Findings (技術的発見)

### 依存関係
- **Python 3.12 が一意に強制される**: `lerobot[smolvla]`(PyPI 0.5.0/0.5.1)が Python>=3.12 必須。RoboStack humble は cp39/cp311/cp312 ビルドのみ(3.10 無し)。両立点は 3.12 のみ。
- upstream の `setuptools<=58.2.0` は **3.12 で破綻**(stdlib distutils 削除、58 は未 vendoring)。`>=68,<80` を採用(`setup.py develop`=`--symlink-install` は setuptools 80 で削除のため <80)。
- lerobot 0.5.x は **numpy 2.x 必須**(`>=2.0,<2.3`)、`packaging<26`(conda-forge 既定 26.x と PyPI ソルバ競合)、lerobot 0.5.1 は opencv<4.14(conda 4.13 OK)。
- **torch 2.10.0+cu128(nightly)は地雷**: `torch._dynamo.utils.NP_SUPPORTED_MODULES` を削除しており torchvision 0.25 が import 失敗。→ **torch 2.7.1+cu126 / torchvision 0.22.1+cu126** に固定で解決。
- RoboStack 経由なので **Focal(20.04)でも動く**(システム ROS 非依存)。CUDA 12.8 ビルド torch も 12.7 ドライバ上で前方互換動作(実 matmul で確認)。

### SmolVLA / lerobot 0.5.1 実 API
- `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`(~865MB, HF キャッシュ)。
- `make_pre_post_processors(config, pretrained_path=path)` が正しいシグネチャ(ChatGPT スケルトンの `preprocessor_overrides` 等は存在しない引数だった)。
- 入力は **un-batched dict**(state, images[0,1], "task"文字列)。preprocessor が batch 次元と言語トークンを付与。
- `select_action` 出力は `Tensor (1,6)` on GPU。postprocess へは `action.as_subclass(PolicyAction)` で cast。
- smolvla_base は **6-DoF**(SO-100 系)。`max_state_dim=32` の flex 構造で **3次元 state のまま推論可能**(pad 不要)。我々の toy(3関節)とは意味的にミスマッチ(=未学習では意味ある挙動は出ない)。
- 期待カメラキーは `observation.images.camera1/2/3`(policy.yaml をこれに合わせた)。

### MuJoCo 描画 / モデル
- `MUJOCO_GL=egl`(activation 既定)で **A5000 上の offscreen 実描画が動く**。`__del__` 時の EGLError は無害(コンテキスト破棄順序)。
- **合成画像の識別法**: 合成は R=列方向ランプ(列内分散≈0)/G=行方向ランプ/B=全画素一定。実描画はこの3条件をすべて破る。
- toy_arm が指令に追従しなかった根因(3重複): ①joint2 の axis `0 1 0` でグリッパが床にめり込み固定 ②隣接リンクのセルフ衝突 ③kp=12 が低く重力に負ける。→ axis 反転、range -0.3..1.3、`<contact><exclude>`、kp=200、damping↑ で追従改善。

---

## Struggles (詰まり・失敗)
- **依存解決が多段で難航**: python 3.10→3.11(lerobot で不可)→3.12、numpy/packaging/opencv/torch-nightly と次々に競合。`pixi lock`(DL 前 solve)で高速反復して突破。
- **ChatGPT スケルトンの SmolVLA コードが実 API と不一致**: import パス・引数・select_action 周りが推測のまま。インストール済みソースを正典として読み直して修正。
- **偽の「動き」事件(重要)**: worker の最初の "動く MP4" は、**/tmp に残存した多重 sim/policy プロセス**が同一トピックへ publish し、カメラ画像が2状態フリッカした**アーティファクト**だった。auditor が「連続フレーム差分は巨大なのに same-parity(gap-2)差分は極小」から看破。クリーン環境では実は**静止**。
- **未学習 SmolVLA の退化挙動**: ほぼ無意味な観測に対し両関節を可動限界へ張り付かせる指令を出し続け、action_scale=0.3 では一瞬で飽和→凍結。
- **同じ轍を繰り返すリスク**: worker が (a)録画前のプロセス全 kill / 単一インスタンス検証、(b)action_scale を下げる、を再三省略し、フリッカ偽陽性/飽和静止を「成功」と誤報告。→ 検証エビデンス(Publisher==1 出力・same-parity 差分・joint 時系列)を**報告必須項目**に格上げし、skills(clean-sim-run/record-and-verify-motion)に手順を固定化して再発防止。

---

## Tips (次に活きる知見)
- **依存は `pixi lock` で solve だけ高速反復** → 通ってから重い `pixi install`。
- **API はインストール済みパッケージのソースを正典に**(web のバージョンドリフトより確実)。
- **ROS 録画の前後に残存プロセスを必ず全 kill**し、`ros2 topic info` の **Publisher count==1** を確認してから録画(多重publishのフリッカ偽陽性を防ぐ)。
- **動きの真偽は same-parity(gap-2)フレーム差分**で判定(連続差分だけ見るとフリッカに騙される)。加えて joint の range>0・net変位/path長比。
- **合成 vs 実描画**は R列/G行ランプ・B一様 で機械判定。
- **standalone スクリプトで最速反復**(SmolVLA 推論・EGL 描画を ROS なしで先に通す)。
- **成果物(MP4/PNG)は repo `temp/` に保存**(/tmp は揮発)。
- 未学習方策で「見せる」には **action_scale を小さく**して可動域をゆっくり横断させる(飽和張り付きを避ける)。

---

## Timeline (節目)
1. zip 解凍 → 目的/要件/DoD 整理 → PC 確認(A5000/Focal/CUDA12.7)。
2. git init、main 温存、`focal-cu127` 作成、Focal/CUDA12.7 適合。
3. **M0**: 依存解決(py3.12 等)→ install → doctor(torch CUDA・A5000)→ build → mock 閉ループ。監査 PASS、commit `ac23bf8`。
4. **推論+可視化フェーズ**: Track1(SmolVLA 推論 standalone)PASS / Track2(EGL 実描画+MP4基盤)PASS / モデル追従修正 PASS。
5. E2E v3: 機構(SmolVLA駆動・実描画)は成立も、**可視運動は多重プロセス汚染の偽物**と判明し FAIL → クリーン録画＋飽和回避で再挑戦中(現在)。
