# BOAT RACE AI v2.8.1

v2.8 の結果自動精算で発生した pandas TypeError を修正。

## 修正内容
- `actual_combo` / `combo` / `status` / `miss_type` 等を object 型へ明示的に正規化
- CSV読込み時に全列の型を再構成
- `actual_combo="1-2-3"` のような文字列結果を安全に保存
- `hit` は True / False / NA を安全に保持
- 払戻・収支列は数値型へ正規化
- 結果自動精算で例外が起きてもアプリ全体を停止しない
- v2.8 の未見データ100%超実戦投入ゲートは維持

GitHubへはZIP内のファイルをそのまま上書きできます。
