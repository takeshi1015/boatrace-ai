# BOAT RACE AI 無料版 v2.1 修正版

修正内容:
- 空のオッズDataFrameでも `combo` / `odds` 列を必ず保持
- `merge()` の KeyError を防止
- 1レースのエラーでアプリ全体が停止しないよう個別例外処理を追加
- 実オッズ未取得時は期待値を推測しない
