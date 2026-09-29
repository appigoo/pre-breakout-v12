# Pre-Rise Lab V3.3

## 核心目標
自動研究每隻股票過去約 5 年，在大幅上升之前通常出現什麼特徵，建立該股票自己的 Stock DNA，再檢查今天是否再次出現類似狀態。

## V3.3 新增
- Stock-specific large-move event definition：用 TRAIN 的未來 MFE 分布學習事件門檻
- Pre-Rise Timeline：T-20 至 T-1
- Feature Detective：爆升事件 vs 普通日的 Effect Size
- Stock DNA：每隻股票獨立選特徵與方向/門檻
- OOS + embargo：DNA 與事件門檻不使用 OOS
- Current State Match：今天與 TRAIN 歷史狀態相似度
- Historical Replay：盲看事件前資料，再 Reveal 未來
- EMA20 extended / reclaim
- Breakout / entry decision 仍保留，但不凌駕核心研究目標
- Model QA panel

## 重要限制
1. 新聞/財報尚未進入歷史 OOS；沒有可靠的 point-in-time 歷史新聞 archive 時，不把今日新聞倒灌到過去。
2. MFE/MAE/future return 只能作 outcome，不能作 live feature。
3. 「BUY NOW」不是價格預測保證；需要 OOS edge + current state + entry quality。
4. V3.3 是研究原型，應先用 TSLA/INTC/RKLB 等多股票驗證，再決定下一步模型。

## 啟動
```bash
pip install -r requirements.txt
streamlit run app.py
```
